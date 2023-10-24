"""Tests for the different TransformModules to make sure tensors are properly registered."""
import io
import itertools
from typing import Any, TypeVar

import pytest
import torch
from tensordict import TensorDictBase, make_tensordict

from causica.distributions.transforms import SequentialTransformModule
from causica.distributions.transforms.base import TransformModule
from causica.distributions.transforms.joint import JointTransformModule


class _OffsetTransformModule(TransformModule[torch.Tensor, torch.Tensor]):
    """Dummy transform module that adds a constant to the input tensor.

    Used for testing the registration of transform modules."""

    def __init__(self, offset: torch.Tensor):
        super().__init__(cache_size=0)
        self.offset: torch.Tensor
        self.register_buffer("offset", offset)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.offset

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        return y - self.offset


def _test_triplets():
    """Generate test triplets of (data, transform, expected_result)."""
    data = torch.randn((3, 1), dtype=torch.float32)
    offset = torch.full((3, 1), 7.5, dtype=torch.float32)
    transform = _OffsetTransformModule(offset)
    return [
        (data, transform, data + offset),
        (data, SequentialTransformModule[torch.Tensor, torch.Tensor](transform, transform.inv), data),
        (data, SequentialTransformModule[torch.Tensor, torch.Tensor](transform, transform), data + 2 * offset),
        (make_tensordict({"a": data}), JointTransformModule({"a": transform}), make_tensordict({"a": data + offset})),
    ]


X = TypeVar("X", torch.Tensor, TensorDictBase)
Y = TypeVar("Y", torch.Tensor, TensorDictBase)


@pytest.mark.parametrize("data,transform,expected_result", _test_triplets())
def test_transform_module_output(data: X, transform: TransformModule[X, Y], expected_result: Y) -> None:
    output = transform(data)
    torch.testing.assert_close(output, expected_result)

    inverse = transform.inv
    assert inverse.inv is transform
    torch.testing.assert_close(inverse(output), data)


@pytest.mark.parametrize("data,transform,_", _test_triplets())
@pytest.mark.parametrize("to_kwargs", [{"dtype": torch.float16}])
def test_registration(data: X, transform: TransformModule[X, Y], _, to_kwargs: dict[str, Any]) -> None:
    """Test that registration is working by testing that the state can be moved and loaded."""
    transform_modified: TransformModule[X, Y] = transform.to(**to_kwargs)

    # Collect parameters and buffers as tensors
    tensors = dict(itertools.chain(transform.named_buffers(), transform.named_parameters()))
    tensors_modified = dict(itertools.chain(transform_modified.named_buffers(), transform_modified.named_parameters()))

    # Check that the tensors are equivalent
    assert set(tensors) == set(tensors_modified)
    for name in tensors:
        torch.testing.assert_close(tensors[name].to(**to_kwargs), tensors_modified[name])

    # Check that the state dict is consistent and picklable
    state_dict = transform_modified.state_dict()
    with io.BytesIO() as f:
        torch.save(state_dict, f)
        f.seek(0)
        state_dict = torch.load(f)
    for name in tensors:
        torch.testing.assert_close(tensors[name].to(**to_kwargs), state_dict[name])

    # Produce the output for x
    if isinstance(data, TensorDictBase):
        x_modified = data.apply(lambda x_: x_.to(**to_kwargs))
    else:
        x_modified = data.to(**to_kwargs)
    y_modified = transform_modified(x_modified)
    y = transform(data)

    # Check that the output remains correct, i.e. the transformation is approx equivariant w.r.t. the `to` operator.
    if isinstance(y, TensorDictBase):
        assert isinstance(y_modified, TensorDictBase)  # plays nicer with mypy than checking type equality
        for key in y.keys():
            torch.testing.assert_close(y_modified.get(key), y.get(key).to(**to_kwargs), atol=2e-2, rtol=1e-2)
    else:
        assert isinstance(y_modified, torch.Tensor)  # plays nicer with mypy than checking type equality
        torch.testing.assert_close(y_modified, y.to(**to_kwargs), atol=2e-2, rtol=1e-2)


def test_transform_module_registration_buffers() -> None:
    # Check that z is in buffers
    offset = torch.randn((5, 1))
    transform = _OffsetTransformModule(offset)
    buffers = dict(transform.named_buffers())
    torch.testing.assert_close(buffers["offset"], offset)


def test_sequential_transform_module_inner_buffers() -> None:
    offset = torch.randn((5, 1))
    transform = _OffsetTransformModule(offset)
    seq_transform = SequentialTransformModule[torch.Tensor, torch.Tensor](transform, transform.inv)
    # Check that buffers are stored for the inner transformation
    seq_buffers = dict(seq_transform.named_buffers())
    for name, buffer in transform.named_buffers():
        torch.testing.assert_close(seq_buffers[f"0.{name}"], buffer)


def test_joint_transform_module_inner_buffers() -> None:
    offset = torch.randn((2,))
    transform = _OffsetTransformModule(offset)
    joint_transform = JointTransformModule({"a": transform})
    # Check that buffers are stored for the inner transformation
    joint_buffers = dict(joint_transform.named_buffers())
    for name, buffer in transform.named_buffers():
        torch.testing.assert_close(joint_buffers[f"transformations.a.{name}"], buffer)
