import pytest
import torch

from causica.distributions import NoiseAccessibleMultivariateNormal


@pytest.mark.parametrize(("batch", "dimension"), [(1, 10), (2, 5)])
def test_init(batch, dimension):
    mean = torch.randn((batch, dimension))
    cov = torch.diag_embed(torch.ones((batch, dimension)))
    noise_model = NoiseAccessibleMultivariateNormal(loc=mean, covariance_matrix=cov)
    assert noise_model.loc.shape == (batch, dimension)
    assert noise_model.covariance_matrix.shape == (batch, dimension, dimension)


def test_sample_to_noise():

    # batch size 1
    obs = torch.Tensor([[10, 15]])
    pred = torch.Tensor([[5, 10]])
    noise_model = NoiseAccessibleMultivariateNormal(loc=pred, covariance_matrix=torch.eye((2)).repeat(2, 1, 1))
    assert torch.allclose(noise_model.sample_to_noise(obs), torch.Tensor([[5, 5]]))

    # no batch size
    obs = torch.Tensor([10, 15])
    pred = torch.Tensor([5, 10])
    noise_model = NoiseAccessibleMultivariateNormal(loc=pred, covariance_matrix=torch.eye((2)).repeat(2, 1, 1))
    assert torch.allclose(noise_model.sample_to_noise(obs), torch.Tensor([5, 5]))

    # batch size 2
    obs = torch.Tensor([[10, 15], [20, 25]])
    pred = torch.Tensor([[5, 10], [18, 19]])
    noise_model = NoiseAccessibleMultivariateNormal(loc=pred, covariance_matrix=torch.eye((2)).repeat(2, 1, 1))
    assert torch.allclose(noise_model.sample_to_noise(obs), torch.Tensor([[5, 5], [2, 6]]))


def test_noise_to_sample():
    # batch size 1
    noise = torch.Tensor([[5, 5]])
    pred = torch.Tensor([[5, 12]])
    noise_model = NoiseAccessibleMultivariateNormal(loc=pred, covariance_matrix=torch.eye((2)).repeat(2, 1, 1))
    assert torch.allclose(noise_model.noise_to_sample(noise), torch.Tensor([[10, 17]]))

    # no batch size
    noise = torch.Tensor([5, 5])
    pred = torch.Tensor([5, 12])
    noise_model = NoiseAccessibleMultivariateNormal(loc=pred, covariance_matrix=torch.eye((2)).repeat(2, 1, 1))
    assert torch.allclose(noise_model.noise_to_sample(noise), torch.Tensor([10, 17]))
    # batch size 2
    noise = torch.Tensor([[5, 5], [2, 6]])
    pred = torch.Tensor([[5, 12], [18, 19]])
    noise_model = NoiseAccessibleMultivariateNormal(loc=pred, covariance_matrix=torch.eye((2)).repeat(2, 1, 1))
    assert torch.allclose(noise_model.noise_to_sample(noise), torch.Tensor([[10, 17], [20, 25]]))
