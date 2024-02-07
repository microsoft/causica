"""Module that provides data normalization functionality."""
from typing import Any, Callable, Optional

import torch
import torch.distributions as td
from tensordict import TensorDictBase
from torch import nn
from torch.distributions import constraints

from causica.distributions.transforms import JointTransformModule, SequentialTransformModule, TransformModule

Normalizer = TransformModule[TensorDictBase, TensorDictBase]
FitNormalizerType = Callable[[TensorDictBase], Normalizer]


class LoadNoneTensorMixin(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._register_load_state_dict_pre_hook(self._update_tensor_size_on_load)

    def _update_tensor_size_on_load(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs) -> None:
        _ = args, kwargs
        for key, value in state_dict.items():
            local_key = key.removeprefix(prefix)
            if hasattr(self, local_key) and getattr(self, local_key) is None:
                setattr(self, local_key, torch.empty_like(value))


class LogTransform(TransformModule[torch.Tensor, torch.Tensor], td.Transform, LoadNoneTensorMixin):

    """
    A transform to apply the log function to a single tensor plus an offset.
    """

    bijective = True
    domain = constraints.greater_than_eq(0)
    codomain = constraints.real

    arg_constraints = {"offset": constraints.greater_than_eq(0)}

    def __init__(self, offset: Optional[torch.Tensor]) -> None:
        """
        Args:
            offset: the offset added to the single tensor
        """
        super().__init__()
        self.offset: torch.Tensor
        self.register_buffer("offset", offset)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x + self.offset)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        return torch.exp(y) - self.offset

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.abs(1.0 / (x + self.offset)))


class Standardizer(TransformModule[torch.Tensor, torch.Tensor], td.AffineTransform, LoadNoneTensorMixin):
    """Standardizer module for a single variable, ie a single tensor."""

    def __init__(self, mean: Optional[torch.Tensor], std: Optional[torch.Tensor], *args, **kwargs) -> None:
        """
        Args:
            mean: Mean of the variable
            std: Standard deviation of the variable
            *args, **kwargs: Passed to the AffineTransform
        """
        loc = scale = None
        if mean is not None and std is not None:
            loc = -mean / std
            scale = 1 / std
        super().__init__(loc, scale, *args, **kwargs)
        del self.loc, self.scale  # Unset these temporarily to allow registering as buffers

        self.loc: torch.Tensor
        self.scale: torch.Tensor
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)


def fit_log_normalizer(
    data: TensorDictBase, default_offset: float = 1.0, min_margin: float = 0.0, keys: Optional[list[str]] = None
) -> Normalizer:
    """Fits a log standardizer to the tensordict.

    The fitted log normalizer computes:
        log(x - min(data) * (min(data) < 0) + min + (max(data) - min(data)) * min_margin + offset).

    Args:
        data: The data to fit the standardizer to.
        default_offset: An additional offset to use. The offset is the min value + default_offset. Must be positive.
        min_margin: Adds a fraction of the range of data to the minimum offset to avoid log(x<=0) on unseen data.
        keys: Limit the set of keys to log transform if set.

    Returns:
        The log standardizer.
    """
    assert default_offset > 0, "default_offset must be positive"
    # For min_value >= 0, offset = default_offset; min_value<0 offset=abs(min_value)+default_offset
    data = data.select(*keys) if keys else data
    min_values = data.apply(lambda x: torch.min(x, dim=0, keepdim=False).values, batch_size=torch.Size())
    max_values = data.apply(lambda x: torch.max(x, dim=0, keepdim=False).values, batch_size=torch.Size())
    offsets = min_values.apply(
        lambda min_, max_: torch.where(
            min_ >= 0, default_offset * torch.ones_like(min_), torch.abs(min_) + default_offset
        )
        + (max_ - min_) * min_margin,
        max_values,
    )
    return JointTransformModule({key: LogTransform(offset) for key, offset in offsets.items()})


def fit_standardizer(data: TensorDictBase, keys: Optional[list[str]] = None) -> Normalizer:
    """Return a standardizer that updates data to zero mean and unit standard deviation."""
    data = data.select(*keys) if keys else data
    means = data.apply(
        lambda x: torch.mean(x, dim=0, keepdim=False),
        batch_size=torch.Size(),
    )
    # Filter out std == 0
    stds = data.apply(
        lambda x: torch.std(x, dim=0, keepdim=False),
        batch_size=torch.Size(),
    ).apply(lambda x: torch.where(x == 0, torch.ones_like(x), x))

    return JointTransformModule({key: Standardizer(means.get(key), stds.get(key)) for key in means.keys()})


def chain_normalizers(*fit_functions: FitNormalizerType) -> FitNormalizerType:
    """Chain a number of normalizers together.

    Args:
        *fit_functions: Functions that produce normalizers.

    Returns:
        A function that fits the sequence of normalizers.
    """

    def sequential_fitting(X: TensorDictBase) -> Normalizer:
        transform_modules = []
        for fit_function in fit_functions:
            transform_module = fit_function(X)
            X = transform_module(X)
            transform_modules.append(transform_module)
        return SequentialTransformModule[TensorDictBase, TensorDictBase](*transform_modules)

    return sequential_fitting


def infer_compatible_log_normalizer_from_checkpoint(state_dict: dict[str, Any]) -> Normalizer:
    """Infers a normalizer compatible with a model checkpoint.

    Assumes that `normalizer` is stored in the toplevel of the checkpoint and that it is a `SequentialTransformModule`,
    and may contain a `LogTransform` and a `Standardizer`. If both are set they are always in that order.

    Args:
        state_dict: The state dict of the checkpoint to load.

    Returns:
        An unitiliazed normalizer.
    """
    # Infer normalizers per variable from the checkpoint
    normalizers_dicts: dict[int, dict[str, TransformModule[torch.Tensor, torch.Tensor]]] = {0: {}, 1: {}}
    for key in state_dict.keys():
        if key.startswith("normalizer."):
            *_, variable_name, state_name = key.split(".")
            if state_name == "offset":
                normalizers_dicts[0][variable_name] = LogTransform(None)
            elif state_name in {"loc", "scale"} and variable_name not in normalizers_dicts[1]:
                normalizers_dicts[1][variable_name] = Standardizer(None, None)

    # Construct the full empty normalizer ready to be initialized from the checkpoint
    joint_transforms = [JointTransformModule(normalizer) for normalizer in normalizers_dicts.values() if normalizer]
    if not joint_transforms:
        return JointTransformModule({})  # No normalizer, return a passhtrough normalizer

    return SequentialTransformModule[TensorDictBase, TensorDictBase](*joint_transforms)
