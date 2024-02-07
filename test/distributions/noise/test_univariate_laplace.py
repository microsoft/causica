import pytest
import torch

from causica.distributions.noise import UnivariateLaplaceNoise


@pytest.mark.parametrize(("batch", "dimension"), [(1, 10), (2, 5)])
def test_init(batch, dimension):
    mean = torch.randn((batch, dimension))
    scale = torch.ones((batch, dimension))
    noise_model = UnivariateLaplaceNoise(loc=mean, scale=scale)
    assert noise_model.loc.shape == (batch, dimension)
    assert noise_model.scale.shape == (batch, dimension)


def test_sample_to_noise():
    # batch size 1
    obs = torch.Tensor([[10, 15]])
    pred = torch.Tensor([[5, 10]])
    noise_model = UnivariateLaplaceNoise(loc=pred, scale=torch.ones_like(pred))
    assert torch.allclose(noise_model.sample_to_noise(obs), torch.Tensor([[5, 5]]))

    # no batch size
    obs = torch.Tensor([10, 15])
    pred = torch.Tensor([5, 10])
    noise_model = UnivariateLaplaceNoise(loc=pred, scale=torch.ones_like(pred))
    assert torch.allclose(noise_model.sample_to_noise(obs), torch.Tensor([5, 5]))

    # batch size 2
    obs = torch.Tensor([[10, 15], [20, 25]])
    pred = torch.Tensor([[5, 10], [18, 19]])
    noise_model = UnivariateLaplaceNoise(loc=pred, scale=torch.ones_like(pred))
    assert torch.allclose(noise_model.sample_to_noise(obs), torch.Tensor([[5, 5], [2, 6]]))


def test_noise_to_sample():
    # batch size 1
    noise = torch.Tensor([[5, 5]])
    pred = torch.Tensor([[5, 12]])
    noise_model = UnivariateLaplaceNoise(loc=pred, scale=torch.ones_like(pred))
    assert torch.allclose(noise_model.noise_to_sample(noise), torch.Tensor([[10, 17]]))

    # no batch size
    noise = torch.Tensor([5, 5])
    pred = torch.Tensor([5, 12])
    noise_model = UnivariateLaplaceNoise(loc=pred, scale=torch.ones_like(pred))
    assert torch.allclose(noise_model.noise_to_sample(noise), torch.Tensor([10, 17]))
    # batch size 2
    noise = torch.Tensor([[5, 5], [2, 6]])
    pred = torch.Tensor([[5, 12], [18, 19]])
    noise_model = UnivariateLaplaceNoise(loc=pred, scale=torch.ones_like(pred))
    assert torch.allclose(noise_model.noise_to_sample(noise), torch.Tensor([[10, 17], [20, 25]]))
