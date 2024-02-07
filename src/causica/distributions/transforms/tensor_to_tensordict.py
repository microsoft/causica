import torch
from tensordict import TensorDict, TensorDictBase
from torch.distributions.constraints import Constraint

from causica.distributions.transforms.base import TypedTransform


class TensorToTensorDictTransform(TypedTransform[torch.Tensor, TensorDictBase]):
    """
    A transform for converting a torch tensor to a TensorDict.

    It extracts the slices from the last dimension of the tensor and assigns them to the correct key.
    """

    arg_constraints: dict[str, Constraint] = {}
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

    def _call(self, x: torch.Tensor) -> TensorDictBase:
        """Create a Tensordict by retrieving the slice associated with each key."""
        return TensorDict({name: x[..., slice_] for name, slice_ in self.slices.items()}, batch_size=x.shape[:-1])

    def _inverse(self, y: TensorDictBase) -> torch.Tensor:
        """
        Create a tensor by stacking the slice associated with each key.

        Args:
            y: Tensordict with batch_shape
        Returns:
            A tensor with shape batch_shape + [output_shape]
        """
        return torch.cat([y[name] for name in self.slices], dim=-1)

    def log_abs_det_jacobian(self, _: torch.Tensor, y: TensorDictBase) -> TensorDictBase:
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

    def order_td(self, td: TensorDictBase) -> TensorDictBase:
        """Order the keys of a TensorDict to match the order of the shapes."""
        return td.select(*self.shapes.keys(), inplace=True)


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
    next_idx = 0
    for name, shape in shapes.items():
        next_idx = idx + shape[-1]
        slices[name] = slice(idx, next_idx)
        idx = next_idx

    return next_idx, slices
