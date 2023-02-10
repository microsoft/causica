from typing import Dict, Hashable, Optional, Union

import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict

DictOfTensors = Union[Dict[Hashable, torch.Tensor], Dict[Hashable, np.ndarray]]


def convert_one_hot(
    data: TensorDict,
    one_hot_sizes: Optional[Dict[Hashable, int]] = None,
):
    """
    Args:
        data: A tensordict representing the underlying data
        one_hot_keys: A list of keys to convert to one hot encodings
    """
    # one hot encode the categorical variables
    if one_hot_sizes is None:
        return data

    new_data = data.clone()
    for key, num_classes in one_hot_sizes.items():
        cat_var = new_data[key]
        assert cat_var.shape == new_data.batch_size + (1,), "Only support 1D categorical values"
        new_data[key] = torch.nn.functional.one_hot(cat_var[..., 0].to(torch.long), num_classes=num_classes)
    return new_data


def tensordict_shapes(tds: TensorDict) -> Dict[Hashable, torch.Size]:
    """Return the shapes within the TensorDict without batch dimensions."""
    return {key: val.shape[len(tds.batch_size) :] for key, val in tds.items()}


def tensordict_from_pandas(df: pd.DataFrame) -> TensorDict:
    """
    Create a `TensorDict` from a pandas dataframe.

    It supports pandas `MultiIndex`s of depth 2 and uses the top level index
    to determine the number of keys in the dictionary.
    """
    if isinstance(df.columns, pd.MultiIndex):
        assert df.columns.nlevels == 2, "Only support MultiIndex of depth 2."
        data = {key: df[key].to_numpy() for key in df.columns.get_level_values(0).unique()}
    else:
        # in the single index case, add a dimension to the series, so they have shape [batch, 1]
        data = {key: df[key].to_numpy()[:, None] for key in df.columns}

    return TensorDict(data, batch_size=torch.Size([len(df)]))
