from typing import Generic, Mapping, TypeVar

import torch
import torch.distributions as td
from tensordict import TensorDictBase
from torch import nn

from causica.distributions.transforms.base import TransformModule, TypedTransform


class JointTransform(TypedTransform[TensorDictBase, TensorDictBase]):
    """A joint transform that applies a different transform to each key in the TensorDict.

    Keys in the input that are not found in the transform are left unchanged.

    This is heavily inspired by the `torch.distributions.transforms.StackTransform` class.
    See https://pytorch.org/docs/stable/distributions.html#torch.distributions.transforms.StackTransform
    """

    def __init__(self, transformations: Mapping[str, td.Transform], cache_size: int = 0):
        """
        Args:
            transformations: A dictionary of transforms, where the keys are the keys in the TensorDict
            cache_size: Size of cache. If zero, no caching is done. If one, the latest single value is cached.
                Only 0 and 1 are supported.
        """
        bad_transformation_types = {type(t) for t in transformations.values() if not isinstance(t, td.Transform)}
        if bad_transformation_types:
            raise TypeError(
                "All transformations must be subtypes of `torch.distributions.Transform`, but the "
                f"following are not: {bad_transformation_types} are not."
            )
        if cache_size:
            transformations = {key: t.with_cache(cache_size) for key, t in transformations.items()}
        super().__init__(cache_size=cache_size)
        self.transformations = transformations

    def _call(self, x: TensorDictBase) -> TensorDictBase:
        return x.clone().update(
            {key: transform(x[key]) for key, transform in self.transformations.items() if key in x.keys()}
        )

    def _inverse(self, y: TensorDictBase) -> TensorDictBase:
        # We cannot use ._inv as pylint complains with E202: _inv is hidden because of `self._inv = None`
        # in td.Transform.__init__

        return y.clone().update(
            {key: transform.inv(y[key]) for key, transform in self.transformations.items() if key in y.keys()}
        )

    def log_abs_det_jacobian(self, x: TensorDictBase, y: TensorDictBase) -> TensorDictBase:
        if set(x.keys()) != set(y.keys()):
            raise ValueError("x and y must have the same keys.")

        if not set(self.transformations.keys()) <= set(x.keys()):
            raise ValueError("All keys in transformations must be in x and y.")

        return x.clone().update(
            {
                key: self.transformations[key].log_abs_det_jacobian(x[key], y[key])
                if key in self.transformations
                else torch.zeros_like(x[key])
                for key in x.keys()
            }
        )

    @property
    def bijective(self):
        return all(t.bijective for t in self.transformations.values())

    @property
    def domain(self):
        return {key: t.domain for key, t in self.transformations.items()}

    @property
    def codomain(self):
        return {key: t.codomain for key, t in self.transformations.items()}


T_co = TypeVar("T_co", bound=nn.Module, covariant=True)


class _TypedModuleDict(Generic[T_co], nn.ModuleDict, Mapping[str, T_co]):
    """Allow a ModuleDict to be interpreted as a mapping."""

    def __hash__(self) -> int:
        return nn.ModuleDict.__hash__(self)


class JointTransformModule(JointTransform, TransformModule[TensorDictBase, TensorDictBase]):
    """Joint transform with TransformModule transformations applied per key to a TensorDict."""

    def __init__(self, transformations: Mapping[str, TransformModule], *args, **kwargs):
        """
        Args:
            transformations: A mapping of transforms, where the keys are the keys in the TensorDict.
            *args, **kwargs: Passed to the JointTransform.
        """
        super().__init__(transformations, *args, **kwargs)
        self.transformations = _TypedModuleDict[TransformModule](transformations)
