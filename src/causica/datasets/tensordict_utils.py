from typing import Iterable, Optional, TypeVar

import pandas as pd
import torch
from tensordict.tensordict import TensorDict, TensorDictBase

TD = TypeVar("TD", bound=TensorDictBase)


def convert_one_hot(
    data: TD,
    one_hot_sizes: Optional[dict[str, int]] = None,
) -> TD:
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


def tensordict_shapes(tds: TensorDictBase) -> dict[str, torch.Size]:
    """Return the shapes within the TensorDict without batch dimensions."""
    return {key: val.shape[len(tds.batch_size) :] for key, val in tds.items(include_nested=True, leaves_only=True)}


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


def identity(x):
    """
    An identity function.

    This is needed to override the default collate function for the dataloaders,
    which limit the datatypes you can return and we have `TensorDict` etc.
    """
    return x


def expand_tensordict_groups(tensordict: TensorDictBase, variable_groups: dict[str, list[str]]) -> TensorDict:
    """Expand a TensorDict with variable groups to include the individual variables.

    This only includes groups that are present in the variable groups dictionary.

    Args:
        tensordict: The TensorDict to expand
        variable_groups: A dictionary mapping group names to lists of variable names

    Returns:
        A new TensorDict with the expanded variables
    """
    if variable_groups.keys() < set(tensordict.keys()):
        missing_keys = set(tensordict.keys()) - variable_groups.keys()
        raise ValueError(f"The provided variable_groups are missing the following keys in {missing_keys}.")
    if not all(
        (len(variable_groups[k]) == v.shape[-1])  # Matching shape
        or (len(variable_groups[k]) == 1)  # Or, allow any shape for unitary variable groups to allow categoricals
        for k, v in tensordict.items()
    ):
        raise ValueError("All variables in tensordict must have the same number of elements as the variable groups.")

    return TensorDict(
        {
            variable_name: tensordict.get(group_name)
            if len(variable_names) == 1
            else tensordict.get(group_name)[..., i, None]
            for group_name, variable_names in variable_groups.items()
            for i, variable_name in enumerate(variable_names)
        },
        batch_size=tensordict.batch_size,
    )


def unbind_values(td: TensorDictBase, dim: int = -1) -> TensorDictBase:
    """Return a new TensorDict with the values unbound along the given dimension.

    Negative dimensions are supported, but only if all values in the tensordict have the same number of axes. As opposed
    to `TensorDict.unbind` this supports unbinding along any dimension, not just the batch dimension.

    Args:
        td: The tensordict to unbind.
        dim: The dimension to unbind along for each value in td. Supports negative indexing, but note that when the
            values of td have different numbers of dimensions, the unbinding will act on different axis. However, if the
            values have different numbers of axes and the negative index refers to a batch dim, the batch dim cannot
            be consistently modified and an IndexError will be raised.

    Returns:
        A new nested tensordict where values unbound along dim get keys by their index.

    Examples:
        >>> td = TensorDict({"a": torch.rand(2, 3, 4), "b": torch.rand(2, 3, 4)})
        >>> unbind_values(td, dim=1)
        TensorDict({"a": TensorDict({0: torch.rand(2, 4), 1: torch.rand(2, 4), 2: torch.rand(2, 4)}), "b": ...})
    """
    num_dims = {len(tensor.shape) for tensor in td.values(include_nested=True, leaves_only=True)}
    min_dims = min(num_dims)
    max_dims = max(num_dims)

    if -min_dims > dim or dim >= min_dims:
        raise IndexError("Cannot unbind values along an axis greater than the smallest number of dims in the values")

    batch_size = td.batch_size
    if dim % min_dims < len(td.batch_size):  # If for at least some tensor, dim addresses into the batch dims
        if dim < 0 and min_dims != max_dims:
            raise IndexError(
                "Cannot unbind values along a batch dim with negative indexing when the values have different numbers "
                "of dims"
            )
        # This works because there will only be a wrap around if dim is negative, which when modifying the batch size
        # is only true when min_dims == max_dims. If it is not negative, then we know that dim < min_dims (first check).
        batch_dim = dim % min_dims
        batch_size = td.batch_size[:batch_dim] + td.batch_size[batch_dim + 1 :]

    return td.apply(
        lambda value: TensorDict({str(i): v for i, v in enumerate(value.unbind(dim=dim))}, batch_size),
        batch_size=batch_size,  # Note: Set the batch size on both the nested and outer tensordict
    )


def unbound_items(td: TensorDictBase, dim: int = -1) -> Iterable[tuple[tuple[str, ...], torch.Tensor]]:
    """Return an iterable over the (key, value) pairs for each tensor unbound along dim.

    Iterates recursively over all tensor slices in the tensordict and adds an additional key indicating the slice index
    for each element along the given dimension to unbind over.

    Example:
        >>> td = TensorDict({"a": torch.ones(2, 3, 4)}, batch_size=2)
        >>> dict(unbound_items(td, dim=0))
        {
            ("a", "0"): torch.ones(3, 4),
            ("a", "1"): torch.rand(3, 4),
        }

    Args:
        td: The tensordict to flatten.
        dim: The dimension to unbind along for each value in td.

    See Also:
        `unbind_values` for more details of how the values are unbound.
    """
    unbound = unbind_values(td, dim=dim)
    # TensorDict.items incorrectly assumes that the keys are always strings, even though they are tuples for nested
    # leaves. Since we're unbinding above, they will always be tuples below. Ignore this perceived type error.
    return unbound.items(include_nested=True, leaves_only=True)  # type: ignore


def expand_td_with_batch_shape(td: TensorDict, batch_shape: torch.Size) -> TensorDict:
    """Expand the tensordict with a given batch shape.

    This is used for making tensors in tensordict broadcastable to a given batch shape.

    We assume that a given tensordict fulfils one of the following conditions:
    1) The tensordict has no batch shape (i.e. `td.batch_dims == 0`) and will be broadcasted to (1, *batch shape) if
        the given batch shape is not empty
    2) The tensordict has a batch shape with a single batch dim (i.e. `td.batch_dims == 1`) and will be broadcasted to
        (-1, *batch shape) if the given batch shape is not empty. -1 indicates no change.
    3) The tensordict has a batch shape with three batch dims (i.e. `td.batch_dims == 3`) and will be broadcasted to
        (-1, *batch shape) if the given batch shape is not empty. -1 indicates no change. If the last dimensions of the
        tensordict batch shape are not broadcastable to the given batch shape, an error will be raised, i.e. if the
        shape is (2, 3, 1) and the given batch shape is (3, 5) the output shape will be (2, 3, 5) but (2, 3, 4) and
        (3, 5) will raise an error.

    This is useful for working with SEMs with batch dims. For example, if you have a SEM with batch shape [2, 3] and
    you want to process a batch of data with batch shape [5] you can use this function to expand the data to have batch
    shape [5, 2, 3] and then apply the SEM to the expanded data. The convention is that the first batch dim is the
    sample dim and the remaining batch dims are the SEM batch dims corresponding to the functional relationships and
    the graph, respectively.

    Args:
        td: a tensordict with batch shape `()` (empty) or `sample_shape` or `sample_shape + batch_shape`
        batch_shape: the batch shape that might be added to the tensordict. The batch shape must have length 0 or 2. A
            length of 0 indicates that the tensordict should not be expanded. A length of 2 indicates that the
            tensordict should be expanded to have 3 batch dims.

    Returns:
        A tensordict with batch shape `sample_shape + batch_shape` with 0, 1, or 3 batch dims.
    """
    if td.batch_dims not in (0, 1, 3):
        raise ValueError(
            f"Cannot expand a tensordict with batch dims {td.batch_dims}. Only batch dims of 0, 1, and 3 are supported."
        )

    batch_dims = len(batch_shape)
    # Return the tensordict if the SEM does not have any batch shape
    if batch_dims == 0:
        return td
    if batch_dims != 2:
        raise ValueError(
            f"Cannot expand a tensordict to batch shape {batch_shape}. Only batch dims of 0 and 2 are supported."
        )

    # Expand the tensordict if the tensordict has no batch shape
    match td.batch_dims:
        case 0:
            # Expand the tensordict from no batch dims to 3 batch dims
            new_shape = (1,) + batch_shape
        case 1:
            # Expand the tensordict from 1 batch dim to 3 batch dims
            td = td[:, None, None]
            new_shape = (td.batch_size[0],) + batch_shape
        case 3:
            # Check that the tensordict batch shape matches the given shape or is broadcastable to the given shape
            if any((td.batch_size[dim] != batch_shape[dim - 1] and td.batch_size[dim] != 1) for dim in (1, 2)):
                raise ValueError(
                    f"Cannot expand a tensordict with batch shape {td.batch_size} to batch shape {batch_shape}."
                )
            new_shape = (td.batch_size[0],) + batch_shape
    return td.expand(*new_shape)
