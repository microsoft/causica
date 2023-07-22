"""
Wrapper around torch.distributions.transforms to allow for joint transforms on TensorDicts.
"""
from typing import Mapping

import torch
import torch.distributions as td
from tensordict import TensorDict


class JointTransform(td.Transform):
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
        assert all(
            isinstance(t, td.Transform) for t in transformations.values()
        ), f"All transformations must be of type torch.distributions.Transform, but are {[type(t) for t in transformations.values()]}."
        if cache_size:
            transformations = {key: t.with_cache(cache_size) for key, t in transformations.items()}
        super().__init__(cache_size=cache_size)

        self.transformations = transformations

    def _call(self, x: TensorDict) -> TensorDict:
        return x.clone().update(
            {key: transform(x[key]) for key, transform in self.transformations.items() if key in x.keys()}
        )

    def _inverse(self, y: TensorDict) -> TensorDict:
        # We cannot use ._inv as pylint complains with E202: _inv is hidden because of `self._inv = None`
        # in td.Transform.__init__

        return y.clone().update(
            {key: transform.inv(y[key]) for key, transform in self.transformations.items() if key in y.keys()}
        )

    def log_abs_det_jacobian(self, x: TensorDict, y: TensorDict) -> torch.Tensor:
        if set(x.keys()) != set(y.keys()):
            raise ValueError("x and y must have the same keys.")

        if not set(self.transformations.keys()).issubset(x.keys()):
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


class TensorToTensorDictTransform(td.Transform):
    """
    A transform for converting a torch tensor to a TensorDict.

    It extracts the slices from the last dimension of the tensor and assigns them to the correct key.
    """

    bijective = True

    def __init__(self, shapes: dict[str, torch.Size]):
        """
        Args:
            shapes: the shapes of each of the keys
        """
        super().__init__()
        self.shapes = shapes
        self.num_keys = len(shapes)
        self.output_shape, self.slices = shapes_to_slices(self.shapes)

    def _call(self, x: torch.Tensor) -> TensorDict:
        """Create a Tensordict by retrieving the slice associated with each key."""
        return TensorDict({name: x[..., slice_] for name, slice_ in self.slices.items()}, batch_size=x.shape[:-1])

    def _inverse(self, y: TensorDict) -> torch.Tensor:
        """
        Create a tensor by stacking the slice associated with each key.

        Args:
            y: Tensordict with batch_shape
        Returns:
            A tensor with shape batch_shape + [output_shape]
        """
        return torch.cat([y[name] for name in self.slices], dim=-1)

    def log_abs_det_jacobian(self, _: torch.Tensor, y: TensorDict) -> TensorDict:
        """This transformation doesn't affect the log det jacobian"""
        return y.apply(torch.zeros_like)

    def stacked_key_masks(self) -> torch.Tensor:
        """
        Create a binary of matrix of where each key is in the tensor.

        Returns:
            A matrix of shape [num_keys, output_shape] with 1 if the index of the tensor
            belongs to the key corresponding to that row
        """
        stacked_key_masks = torch.zeros((self.num_keys, self.output_shape), dtype=torch.float)
        for i, slice_ in enumerate(self.slices.values()):
            stacked_key_masks[i, slice_] = 1.0
        return stacked_key_masks


def shapes_to_slices(shapes: dict[str, torch.Size]) -> tuple[int, dict[str, slice]]:
    """
    Convert a dictionary of shapes to a dictionary of masks by stacking the shapes

    Each mask corresponds to the embedded location in the tensor

    Args:
        shapes: A dict of key names to shapes
    Returns:
        The shape of the stacked tensor and a dictionary of each key to the mask
    """
    assert all(len(shape) == 1 for shape in shapes.values())

    slices: dict[str, slice] = {}
    idx = 0
    for name, shape in shapes.items():
        next_idx = idx + shape[-1]
        slices[name] = slice(idx, next_idx)
        idx = next_idx

    return next_idx, slices
