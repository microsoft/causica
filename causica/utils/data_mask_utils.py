from typing import List

import numpy as np
import torch

from ..datasets.variables import Variables


def sample_inducing_points(data, mask, row_count):

    # Randomly select inducing points to use to impute data.
    random_inducing_points_location = np.random.choice(data.shape[0], size=row_count, replace=True)
    inducing_data = data[random_inducing_points_location, :]
    inducing_mask = mask[random_inducing_points_location, :]

    return inducing_data, inducing_mask


def add_to_mask(variables: Variables, mask: torch.Tensor, idxs: List[int]) -> torch.Tensor:
    """Add observations to whole columns of a processed input mask for the variables with idxs given in `idxs`."""
    cols_set = set()
    for var_idx in idxs:
        cols = variables.processed_cols[var_idx]
        cols_set.update(cols)
    cols = list(cols_set)

    new_mask = mask.clone()
    new_mask[:, idxs] = 1
    return new_mask


def add_to_data(variables: Variables, data: torch.Tensor, new_vals: torch.Tensor, idxs: List[int]) -> torch.Tensor:
    """Update columns of processed data `data` with values from `new_vals` for the variables with idxs given in `idxs`"""
    cols_set = set()
    for var_idx in idxs:
        cols = variables.processed_cols[var_idx]
        cols_set.update(cols)
    cols = list(cols_set)

    new_data = data.clone()
    new_data[:, cols] = new_vals[:, cols]
    return new_data


def restore_preserved_values(
    variables: Variables,
    data: torch.Tensor,  # shape (batch_size, input_dim)
    imputations: torch.Tensor,  # shape (num_samples, batch_size, input_dim)
    mask: torch.Tensor,  # shape (batch_size, input_dim)
) -> torch.Tensor:  # shape (num_samples, batch_size, input_dim)
    """
    Replace values in imputations with data where mask is True

    """
    assert data.dim() == 2

    assert data.shape == mask.shape
    assert isinstance(data, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    masked_data = data * mask

    # Remove metadata dims to get shape (1, batch_size, output_dim)
    if variables.has_auxiliary:
        variables = variables.subset(list(range(0, variables.num_unprocessed_non_aux_cols)))
        output_var_idxs = torch.arange(variables.num_processed_cols, device=mask.device)
        output_mask = torch.index_select(mask, dim=1, index=output_var_idxs)
        masked_imputations = imputations * (1 - output_mask)

        masked_data = torch.index_select(masked_data, dim=1, index=output_var_idxs)

    else:
        masked_imputations = imputations * (1 - mask)

    # pytorch broadcasts by matching up trailing dimensions
    # so it is OK that masked_imputations.ndim==3 and masked_data.dim==2
    return masked_imputations + masked_data
