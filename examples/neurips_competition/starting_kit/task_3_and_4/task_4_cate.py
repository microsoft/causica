import logging
import os
from typing import List

import numpy as np
import pandas as pd
from numpy.random import default_rng

from causica.models.deci.fold_time_deci import FoldTimeDECI
from examples.neurips_competition.starting_kit.task_2.task_2_util import load_FT_DECI_model
from examples.neurips_competition.starting_kit.task_3_and_4.task_4_util import compute_CATE, get_parser

logger = logging.getLogger(__name__)
log_format = "%(asctime)s %(filename)s:%(lineno)d[%(levelname)s]%(message)s"
logging.basicConfig(level=logging.INFO, force=True, format=log_format)


def compute_CATE_task4(
    model: FoldTimeDECI,
    overall_data: dict,
    conditioning_sample_interval: List[float],
    intervention_value: int,
    reference_value: int,
    effect_idx: int,
    max_student: int = 5,
) -> np.ndarray:
    """
    This function computes the CATE for task 4. The idea is similar to task 2 CATE computation. The main difference is that we
    do not have the explicit conditioning samples. Instead, we extract the conditioning samples from the training data (overall_data)
    and use ratio in conditioning_sample_interval to determine which timestamp we use for conditioning. For each conditioning (timestamp and student),
    we compute the CATE similar to task 2. In the end, we average take average over timestamp and students. The number of student is determined by max_student.
    It will randomly sample the max_student number of students.

    Args:
        model: The trained model.
        overall_data: it is a dictionary. The key is the student id, and the value is ndarray containing the corresponding time series.
        second dimension is the bot action.
        conditioning_sample_interval: A list of float between [0,1] to determine the timestamp.
        intervention_value: The treatment construct, this should be the new construct id after mapping.
        reference_value: The control construct, this should be the new construct id after mapping.
        effect_idx: The target construct id, this should be the new construct id after mapping.
        max_student: The maximum number of students we compute cate for.

    Returns:
        CATE_estimate: numpy ndarray contain the estimate
    """
    cate_list = []
    rng = default_rng()
    select_keys = rng.permutation(np.array(list(overall_data.keys())))[:max_student]

    for key_id in select_keys:
        student_data = overall_data[key_id]
        total_length = student_data.shape[0]
        logger.info(f"Computing CATE for student {key_id}")
        student_cate_list = []
        for ratio in conditioning_sample_interval:
            logger.info(f"For ratio {ratio}")
            cur_length = int(total_length * ratio)
            conditioning_data = student_data[:cur_length, :]
            cur_cate = compute_CATE(
                model, conditioning_data, intervention_value, reference_value, effect_idx, effect_time=1
            )
            student_cate_list.append(cur_cate)
        student_cate = np.array(student_cate_list).mean()
        cate_list.append(student_cate)

    return np.array(cate_list).mean()


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load the trained model
    logger.info("Loading the trained model")
    model = load_FT_DECI_model(model_path=args.model_dir, model_id=args.model_id, device=args.device)

    # Load the questionnaire
    questionnaire = pd.read_csv(args.question_path, index_col=False)
    # Load the construct mapping
    construct_map = np.load(args.construct_map, allow_pickle=True).item()
    # Process the training data into dict
    # Load the processed training data
    logger.info("Loading the processed training data")
    train_data = pd.read_csv(args.train_data, index_col=False, header=None)
    data_dict = {int(student_id): df.to_numpy()[:, 1:] for student_id, df in train_data.groupby(0)}

    cate_list = []
    for query_id in range(len(questionnaire)):
        logger.info(f"Computing CATE for {query_id} query")
        _, effect_idx, treatment_idx, control_idx, _ = questionnaire.loc[query_id]
        # Convert to new construct id
        effect_idx, treatment_idx, control_idx = (
            construct_map[effect_idx] + 1,
            construct_map[treatment_idx] + 1,
            construct_map[control_idx] + 1,
        )
        # The value [0.3, 0.7] can be changed. This specifies at which location we extract the conditioning samples for cate estimation.
        # [0.3, 0.7] means at 30% and 70% of the total length. Adding more values increase the robustness but slows down the computation.
        cur_cate = compute_CATE_task4(
            model, data_dict, [0.3, 0.7], treatment_idx, control_idx, effect_idx, max_student=10
        )
        cate_list.append(cur_cate)

    cate_estimate = np.array(cate_list)  # [num_queries]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Save the cate estimate
    np.save(os.path.join(args.output_dir, "cate_estimate.npy"), cate_estimate)


if __name__ == "__main__":
    main()
