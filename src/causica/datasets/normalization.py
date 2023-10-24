"""Module that provides data normalization functionality."""
from typing import Callable, Optional

import torch
import torch.distributions as td
from tensordict import TensorDictBase
from torch import nn

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


def fit_standardizer(data: TensorDictBase) -> Normalizer:
    """Return a standardizer that updates data to zero mean and unit standard deviation."""
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
