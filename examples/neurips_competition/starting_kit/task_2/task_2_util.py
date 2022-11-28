import argparse
import os
from typing import List, Optional, Tuple, Union

import torch

from causica.models.deci.fold_time_deci import FoldTimeDECI
from causica.models_factory import load_model
from causica.utils.helper_functions import convert_dict_of_lists_to_ndarray
from causica.utils.io_utils import read_json_as


def get_parser() -> argparse.ArgumentParser:
    """
    This will return the argument parser.
    Returns:
        Returned Argument parser
    """
    parser = argparse.ArgumentParser(
        description="Competition Task 2 CATE estimation", formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model_dir", "-md", type=str, help="path to the saved model directory", default="runs/")
    parser.add_argument("--model_id", "-mi", type=str, help="The id of the saved model")
    parser.add_argument(
        "--data_dir",
        "-dd",
        type=str,
        help="path to the directory that contains the intervention/reference/effect pairs",
    )
    parser.add_argument(
        "--data_name",
        "-dn",
        type=str,
        help="the json file name for the intervention/reference/effect pairs",
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, help="path to store the CATE estimation npy file", default="outputs/"
    )
    parser.add_argument(
        "--device", "-dv", default="cpu", help="Name (e.g. 'cpu', 'gpu') or ID (e.g. 0 or 1) of device to use."
    )
    return parser


def load_FT_DECI_model(model_path: str, model_id: str, device: Union[str, int]) -> FoldTimeDECI:
    """
    This will load the trained FT_DECI model specified by the path.
    Args:
        path: path to the saved model
        model_id: model id of the saved model.
    Returns:
        Fold-time DECI model
    """
    model_path_id = os.path.join(model_path, model_id)
    model = load_model(model_id=model_id, models_dir=model_path_id, device=device)
    assert isinstance(model, FoldTimeDECI), "The loaded model is not an instance of FoldTimeDECI model."
    return model


def load_interventions(path: str, file_name: str) -> List[dict]:
    """
    This will load the intervention files specified by path/file_name. Since each intervention json contains a list of dictionaries.
    For competition, the length of the list is 10, meaning 10 CATE queries.
    Each dictionary contains the following keys-value: (1) "conditioning": conditioning samples for the current CATE query;
    (2) "effect_mask": A binary matrix specifying the target variable; (3) "intervention": A matrix specifying which variable to be intervened, and its value;
    (4) "reference": A matrix specifying the reference variable and value.
    Args:
        path: path to the stored intervention files.
        file_name: the name of the stored intervention files.
    Returns:
        A list of dictionaries containing the information of the CATE queries.
    """

    intervention_file = os.path.join(path, file_name)
    raw_intervention_list = read_json_as(intervention_file, list)
    intervention_list = [convert_dict_of_lists_to_ndarray(d) for d in raw_intervention_list]
    return intervention_list


def validate_model(model: FoldTimeDECI, intervention_list: List[dict]):
    """
    This will test if the loaded model is compatible with the intervention list.
    Args:
        model: the model to be validated.
        intervention_list: the list of intervention dictionaries.
    """
    # Assertions to check dimensions
    model_dim = len(model.variables_orig)
    intervention_dim = intervention_list[0]["intervention"].shape[1]
    conditioning_dim = intervention_list[0]["conditioning"].shape[1]
    assert model_dim == intervention_dim, "The model dim is not compatible with the intervention data dimensions."
    assert model_dim == conditioning_dim, "The model dim is not compatible with the conditioning data dimensions."

    # assert if the lag is loaded correctly
    assert (
        model_dim % (model.lag + 1) == 0
    ), f"The number of variables ({model_dim}) of the model should be divisible by the lag + 1 ({model.lag+1})."


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
