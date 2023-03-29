import pytest
import torch
from tensordict import TensorDict
from torch.distributions import AffineTransform

from causica.distributions.transforms import JointTransform


@pytest.fixture(name="affine_transform")
def fixture_affine_transform() -> JointTransform:
    transformation_dict = {
        "x": AffineTransform(loc=torch.tensor([[0.0]]), scale=torch.tensor([[1.0]])),
        "y": AffineTransform(loc=torch.tensor([[1.0]]), scale=torch.tensor([[2.0]])),
    }

    return JointTransform(transformation_dict)


def test_joint_transform_call_and_inv(affine_transform):  #
    data = TensorDict({"x": torch.randn((100, 1)), "y": torch.randn((100, 1))}, batch_size=100)

    transformed_data = affine_transform(data)

    assert torch.allclose(transformed_data["x"], data["x"], atol=1e-6)
    assert torch.allclose(transformed_data["y"], data["y"] * 2 + 1, atol=1e-6)

    assert torch.allclose(affine_transform.inv(transformed_data)["x"], data["x"], atol=1e-6)
    assert torch.allclose(affine_transform.inv(transformed_data)["y"], data["y"], atol=1e-6)


def test_joint_transform_log_abs_det_jacobian(affine_transform):
    data = TensorDict({"x": torch.randn((100, 1)), "y": torch.randn((100, 1))}, batch_size=100)

    transformed_data = affine_transform(data)

    log_abs_det_jacobian = affine_transform.log_abs_det_jacobian(data, transformed_data)

    assert torch.allclose(log_abs_det_jacobian["x"], torch.zeros_like(log_abs_det_jacobian["x"]), atol=1e-6)
    assert torch.allclose(
        log_abs_det_jacobian["y"], torch.log(torch.ones_like(log_abs_det_jacobian["y"]) * 2), atol=1e-6
    )
