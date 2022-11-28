import logging
import os
from typing import Union

import numpy as np
import torch

from causica.models.deci.fold_time_deci import FoldTimeDECI

from .task_2_util import (
    convert_to_FT_DECI_intervention,
    get_parser,
    load_FT_DECI_model,
    load_interventions,
    validate_model,
)

logger = logging.getLogger(__name__)
log_format = "%(asctime)s %(filename)s:%(lineno)d[%(levelname)s]%(message)s"
logging.basicConfig(level=logging.INFO, force=True, format=log_format)


def compute_CATE(
    model: FoldTimeDECI,
    conditioning_sample: Union[np.ndarray, torch.Tensor],
    intervention_value: int,
    reference_value: int,
    effect_idx: int,
    effect_time: int = 2,
) -> np.ndarray:
    """
    This function computes the CATE for FT-DECI, specified by the conditioning samples, intervention value, reference value, and effect index.
    Since we can only intervene on the bot action (i.e. fixed intervened/reference idx), therefore, only the intervention/reference values are required.
    Since it deals with temporal data, we need to specify the ahead-time of the target variable (i.e effect_time). E.g. 2 means the target variable is 2 time step ahead of
    the current time. Since Fold-time DECI is uses a fold-time trick to handle temporal data. In nature, it is still a static model.
    So the maximized effect_time should not be <= model lag, otherwise it won't
    be able to estimate the treatment effect.
    Args:
        model: The model used to estimate the CATE.
        conditioning_sample: ndarray or torch.Tensor with shape [conditioning_history_length, number_variables].
        intervention_value: an integer specifying the intervened topic number. The intervention is always done at the bot action variable.
        reference_value: Reference topic number.
        effect_idx: The target topic number.
        effect_time: The ahead time of the effect.

    Returns:
        CATE estimation
    """

    assert model.lag >= effect_time, "The effect_time should be smaller or equal to the model lag."
    num_variables = conditioning_sample.shape[1]

    if isinstance(conditioning_sample, np.ndarray):
        conditioning_sample = torch.as_tensor(conditioning_sample, dtype=torch.float).to(model.device)

    # Estimate CATE
    # Fill the values at time t
    proc_conditioning_sample = conditioning_sample[-model.lag :, ...]  # [model.lag, num_variables]
    intervention_idx_cond, intervention_value_cond = convert_to_FT_DECI_intervention(
        model_lag=model.lag, conditioning_samples=proc_conditioning_sample
    )
    samples_at_t = model.sample(
        Nsamples=1,
        most_likely_graph=True,
        intervention_idxs=intervention_idx_cond,
        intervention_values=intervention_value_cond,
        samples_per_graph=1,
    )  # [Nsamples, num_variables*(lag+1)]

    # Sample the effect variables
    intervention_t_value = torch.as_tensor(intervention_value).to(model.device)
    ref_t_value = torch.as_tensor(reference_value).to(model.device)
    CATE_list = []
    for cur_t_sample in samples_at_t:
        cur_t_sample = cur_t_sample.view(model.lag + 1, -1)[-1, :]  # [num_variables]
        # CATE with intervention
        cur_intervention_idx, cur_intervention_value = convert_to_FT_DECI_intervention(
            model_lag=model.lag,
            conditioning_samples=cur_t_sample[None, ...].clone(),
            update_intervention_idx=0,
            update_intervention_value=intervention_t_value,
            update_intervention_time=-effect_time,
        )
        cur_int_sample_for_CATE = model.sample(
            Nsamples=100,
            most_likely_graph=True,
            intervention_idxs=cur_intervention_idx,
            intervention_values=cur_intervention_value,
            samples_per_graph=100,
        )  # [Nsamples, num_variables*(lag+1)]

        # CATE with reference
        cur_ref_idx, cur_ref_value = convert_to_FT_DECI_intervention(
            model_lag=model.lag,
            conditioning_samples=cur_t_sample[None, ...].clone(),
            update_intervention_idx=0,
            update_intervention_value=ref_t_value,
            update_intervention_time=-effect_time,
        )
        cur_ref_sample_for_CATE = model.sample(
            Nsamples=100,
            most_likely_graph=True,
            intervention_idxs=cur_ref_idx,
            intervention_values=cur_ref_value,
            samples_per_graph=100,
        )  # [Nsamples, num_variables*(lag+1)]
        # extract the effect variables and compute CATE
        cur_CATE = (
            cur_int_sample_for_CATE[:, effect_time * num_variables + effect_idx]
            - cur_ref_sample_for_CATE[:, effect_time * num_variables + effect_idx]
        ).mean()
        CATE_list.append(cur_CATE.cpu().data.numpy())

    return np.array(CATE_list).mean()


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load the trained model
    logger.info("Loading the trained model")
    model = load_FT_DECI_model(model_path=args.model_dir, model_id=args.model_id, device=args.device)
    # Load the intervention list
    logger.info("Loading the intervention json files")
    intervention_list = load_interventions(path=args.data_dir, file_name=args.data_name)

    # Assertions to check if model is valid
    logger.info("Validating the loaded model")
    validate_model(model=model, intervention_list=intervention_list)

    # Estimating the CATE
    CATE_list = []
    for int_dict in intervention_list:
        logger.info(f"Estimating the CATE for query number {len(CATE_list)}")
        int_value = int(int_dict["intervention"][0, 0])
        ref_value = int(int_dict["reference"][0, 0])
        effect_idx = int(np.where(int_dict["effect_mask"])[1])
        cur_CATE = compute_CATE(
            model=model,
            conditioning_sample=int_dict["conditioning"],
            intervention_value=int_value,
            reference_value=ref_value,
            effect_idx=effect_idx,
            effect_time=2,
        )
        CATE_list.append(cur_CATE)

    CATE_estimate = np.array(CATE_list)

    # Create the output folder if not exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # Save the CATE output
    np.save(os.path.join(args.output_dir, f"cate_estimate_{args.data_name.split('.')[0]}.npy"), CATE_estimate)


if __name__ == "__main__":
    main()
