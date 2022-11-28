import argparse
import copy
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from causica.models.deci.fold_time_deci import FoldTimeDECI


def get_parser_proc_data() -> argparse.ArgumentParser:
    """
    This will return the argument parser.
    Returns:
        Returned Argument parser
    """
    parser = argparse.ArgumentParser(
        description="Competition Task 4 CATE estimation", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--data_path", "-dp", type=str, help="path to the saved raw training file")
    parser.add_argument("--save_dir", "-sd", type=str, help="path to save the processed csv data")

    return parser


def get_parser() -> argparse.ArgumentParser:
    """
    This will return the argument parser.
    Returns:
        Returned Argument parser
    """
    parser = argparse.ArgumentParser(
        description="Competition Task 4 CATE estimation", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model_dir", "-md", type=str, help="path to the saved model directory", default="runs/")
    parser.add_argument("--model_id", "-mi", type=str, help="The id of the saved model")
    parser.add_argument(
        "--question_path",
        "-dd",
        type=str,
        help="the file path tp the construct questionnaire",
    )
    parser.add_argument(
        "--construct_map",
        "-cm",
        type=str,
        help="the npy file for the construct map generated during processing the data",
    )
    parser.add_argument(
        "--train_data",
        "-td",
        type=str,
        help="the csv file to the processed training data",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, help="path to store the CATE estimation npy file", default="outputs/"
    )
    parser.add_argument(
        "--device", "-dv", default="cpu", help="Name (e.g. 'cpu', 'gpu') or ID (e.g. 0 or 1) of device to use."
    )
    return parser


def load_and_process_eedi_data(data_path: str, save_dir: str):
    """
    This will load the csv training data and process it into a format that is similar to synthetic data. Since the raw construct id is not continuous,
    we need to prcess them into new consecutive construct ids. This mapping will be a dict named const_map, i.e. new_const_id = const_map[old_const_id].
    Then the processed data will be saved to as a csv file in save_dir.
    The new construct ID to old construct ID mapping dict will also be save to save_dir.
    Args:
        data_path: The path to the csv file containing the training data.
        save_dir: The path to the directory where the processed data will be saved.
    """
    loaded_data = (
        pd.read_csv(data_path, index_col=False)
        .sort_values(["UserId", "QuizSessionId", "Timestamp"], axis=0)
        .reset_index(drop=True)
    )
    # remove the length<10 users
    _, idx, counts = np.unique(np.array(loaded_data["UserId"]), return_counts=True, return_index=True)
    list_idx_to_remove_nest = [list(range(ind, ind + counts[counts < 10][c])) for c, ind in enumerate(idx[counts < 10])]
    list_idx_to_remove = [v for sublist in list_idx_to_remove_nest for v in sublist]  # flatten it
    proc_loaded_data = loaded_data.drop(labels=list_idx_to_remove, axis=0).reset_index(drop=True)
    # Build construct mapping from the old construct_id to the new construct_id
    const_map = {old_id: new_id for new_id, old_id in enumerate(np.unique(proc_loaded_data["ConstructId"]))}
    # Group the user by user_id
    proc_data_user_group = [df for _, df in proc_loaded_data.groupby("UserId", as_index=False)]

    # Process the data for each user
    proc_final_data: List[float] = []

    for user_data in proc_data_user_group:
        cur_user = user_data["UserId"].iloc[0]
        cur_knowledge = {0: 0.25 * np.ones(len(const_map))}

        const_id = np.unique(user_data["ConstructId"])
        cur_question_number = {
            c_id: len(np.unique(user_data[user_data["ConstructId"] == c_id]["QuestionId"])) for c_id in const_id
        }
        # iterate through each row to process the raw data
        new_time = 0
        cur_bot_list = []
        for row_id, row in user_data.iterrows():
            # get new construct_id
            cur_const_id = row["ConstructId"]
            cur_const_new_id = const_map[cur_const_id]
            if row_id == 0:
                # every row should start with a Checkin question
                assert row["Type"] == "Checkin"
            if row["IsCorrect"] == 1:
                # If a question is correctly answered, no learning happens and it reveals the knowledge of the associated
                # construct. Thus, update the knowledge of the current time step.
                cur_knowledge[new_time][cur_const_new_id] += 0.75 / cur_question_number[cur_const_id]
            elif row["IsCorrect"] == 0:
                # If a queston is incorrectly answered, either a lesson or a hint are given. Thus, learning happens.
                # However, this learning only happens after a Checkin question, since other types (e.g. Checkout, CheckinRetry,etc.)
                # teach the same construct knowledge, we aggregate them into one learning session.
                if row["Type"] == "Checkin":
                    cur_bot_list.append(cur_const_new_id)
                    # Generate new timestamp
                    new_time += 1
                    cur_knowledge[new_time] = copy.deepcopy(cur_knowledge[new_time - 1])

        # Generate the processed data
        if len(cur_bot_list) < len(cur_knowledge):
            # Make them compatible shape
            cur_bot_list.append(0)
        elif len(cur_bot_list) > len(cur_knowledge):
            raise ValueError("The knowledge dict should not be smaller than bot action list")
        if len(cur_bot_list) >= 10:
            # remove too short length data
            cur_data_entry = [[cur_user] + [cur_bot_list[k]] + v.tolist() for k, v in cur_knowledge.items()]
            proc_final_data = proc_final_data + cur_data_entry

    # Save the npy file
    proc_final_data_np = np.array(proc_final_data)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savetxt(os.path.join(save_dir, "train.csv"), proc_final_data_np, delimiter=",")
    # save const_map
    np.save(os.path.join(save_dir, "const_map.npy"), const_map)  # type: ignore


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
            Nsamples=3,
            most_likely_graph=True,
            intervention_idxs=cur_intervention_idx,
            intervention_values=cur_intervention_value,
            samples_per_graph=3,
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
            Nsamples=3,
            most_likely_graph=True,
            intervention_idxs=cur_ref_idx,
            intervention_values=cur_ref_value,
            samples_per_graph=3,
        )  # [Nsamples, num_variables*(lag+1)]
        # extract the effect variables and compute CATE
        cur_CATE = (
            cur_int_sample_for_CATE[:, effect_time * num_variables + effect_idx]
            - cur_ref_sample_for_CATE[:, effect_time * num_variables + effect_idx]
        ).mean()
        CATE_list.append(cur_CATE.cpu().data.numpy())

    return np.array(CATE_list).mean()


def convert_to_FT_DECI_intervention(
    model_lag: int,
    conditioning_samples: torch.Tensor,
    update_intervention_idx: Optional[int] = None,
    update_intervention_value: Optional[torch.Tensor] = None,
    update_intervention_time: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This will convert the conditioning samples to the intervention idx and values for the FT-DECI model. This is used for CATE computation.
    The additional update_intervention_idx and update_intervention_value are used to specify the extra intervention variables at update_intervention_time,
    in addition to the conditioning samples.
    E.g. model.lag = 2, conditioning_samples_length = 2, then the all variables at t-2, t-1 are intervened. Since FT-DECI is a static model, we need to convert this to proper intervention idx.
    If we further set update_intervention_idx = 0, update_intervention_value = 45, update_intervention_time = -2, then we need to update the intervened variable 0 at the time t-2 with value 45.
    Args:
        model_lag: the lag of the model.
        conditioning_samples: The Tensor with shape [conditioning_history_length, num_variables].
        update_intervention_idx: The additional intervention variable idx at intervention_time
        update_intervention_value: The corresponding value of the additional intervention variables.
        update_intervenion_time: a non-positive number specifying the time lag, 0 means the current time, -1 means t-1, etc.

    Returns:
        The intervention idxs that can be used in FT-DECI
        The corresponding intervention values
    """
    if update_intervention_time is not None:
        assert update_intervention_time <= 0, "The intervention time should be a non-positive number."
    _, num_variables = conditioning_samples.shape
    intervention_value = torch.flatten(conditioning_samples)
    intervention_idx = torch.arange(intervention_value.shape[0])

    if (
        update_intervention_idx is not None
        and update_intervention_value is not None
        and update_intervention_time is not None
    ):
        if update_intervention_value.dim() == 0:
            update_intervention_value = update_intervention_value[None, ...]
        update_intervention_value = update_intervention_value.type(intervention_value.dtype)
        # update the intervention with the input intervention_idx and intervention_value
        tot_variables = (model_lag + 1) * num_variables
        update_idx = tot_variables - (abs(update_intervention_time) + 1) * num_variables + update_intervention_idx

        if update_idx in intervention_idx:
            intervention_value[intervention_idx == update_idx] = update_intervention_value
        else:
            intervention_idx = torch.cat(
                [
                    intervention_idx,
                    torch.as_tensor([update_idx]).to(device=intervention_idx.device, dtype=intervention_idx.dtype),
                ],
                dim=0,
            )
            intervention_value = torch.cat([intervention_value, update_intervention_value], dim=0)

    return intervention_idx, intervention_value
