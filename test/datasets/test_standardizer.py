import torch
from tensordict import TensorDict

from causica.datasets.normalization import fit_standardizer


def test_standardizer():
    data = TensorDict(
        {
            "x": torch.randn((100, 3)) * torch.arange(1, 4).float(),
            "y": torch.randn((100, 3)) + torch.arange(1, 4).float(),
        },
        batch_size=100,
    )

    standardizer = fit_standardizer(data)

    standardized_data = standardizer(data)

    # Test mean and std
    assert torch.allclose(
        standardized_data["x"].mean(dim=0), torch.zeros_like(standardized_data["x"].mean(dim=0)), atol=1e-6
    )
    assert torch.allclose(
        standardized_data["y"].mean(dim=0), torch.zeros_like(standardized_data["y"].mean(dim=0)), atol=1e-6
    )

    assert torch.allclose(
        standardized_data["x"].std(dim=0), torch.ones_like(standardized_data["x"].std(dim=0)), atol=1e-6
    )
    assert torch.allclose(
        standardized_data["y"].std(dim=0), torch.ones_like(standardized_data["y"].std(dim=0)), atol=1e-6
    )

    # Test specific values
    assert torch.allclose(
        standardized_data["x"],
        (data["x"] - data["x"].mean(dim=0, keepdim=True)) / data["x"].std(dim=0, keepdim=True),
        atol=1e-6,
    )
    assert torch.allclose(
        standardized_data["y"],
        (data["y"] - data["y"].mean(dim=0, keepdim=True)) / data["y"].std(dim=0, keepdim=True),
        atol=1e-6,
    )

    # Test Inverse
    assert torch.allclose(standardizer.inv(standardized_data)["x"], data["x"], atol=1e-6)
    assert torch.allclose(standardizer.inv(standardized_data)["y"], data["y"], atol=1e-6)


def test_standardizer_subset():
    data = TensorDict(
        {
            "x": torch.randn((100, 3)) * torch.arange(1, 4).float(),
            "y": torch.randn((100, 3)) + torch.arange(1, 4).float(),
        },
        batch_size=100,
    )

    standardizer = fit_standardizer(data.select("x"))

    standardized_data = standardizer(data)
    assert torch.allclose(
        standardized_data["x"],
        (data["x"] - data["x"].mean(dim=0, keepdim=True)) / data["x"].std(dim=0, keepdim=True),
    )
    assert torch.allclose(standardized_data["y"], data["y"])

    # Test mean and std
    assert torch.allclose(
        standardized_data["x"].mean(dim=0), torch.zeros_like(standardized_data["x"].mean(dim=0)), atol=1e-6
    )

    assert torch.allclose(
        standardized_data["x"].std(dim=0), torch.ones_like(standardized_data["x"].std(dim=0)), atol=1e-6
    )


def test_standardizer_with_zero_std():
    data = TensorDict(
        {
            "x": torch.ones((100, 3)),
        },
        batch_size=100,
    )

    standardizer = fit_standardizer(data)

    standardized_data = standardizer(data)

    assert torch.allclose(standardized_data["x"], torch.zeros_like(data["x"]))

    assert torch.allclose(standardizer.inv(standardized_data)["x"], data["x"], atol=1e-6)
