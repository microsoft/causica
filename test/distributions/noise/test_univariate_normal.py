import pytest
import torch

from causica.distributions.noise import UnivariateNormalNoise, UnivariateNormalNoiseModule


@pytest.mark.parametrize(("batch", "dimension"), [(1, 10), (2, 5)])
def test_init(batch, dimension):
    mean = torch.randn((batch, dimension))
    scale = torch.ones((batch, dimension))
    noise_model = UnivariateNormalNoise(loc=mean, scale=scale)
    assert noise_model.loc.shape == (batch, dimension)
    assert noise_model.scale.shape == (batch, dimension)


def test_sample_to_noise():

    # batch size 1
    obs = torch.Tensor([[10, 15]])
    pred = torch.Tensor([[5, 10]])
    noise_model = UnivariateNormalNoise(loc=pred, scale=torch.ones_like(pred))
    assert torch.allclose(noise_model.sample_to_noise(obs), torch.Tensor([[5, 5]]))

    # no batch size
    obs = torch.Tensor([10, 15])
    pred = torch.Tensor([5, 10])
    noise_model = UnivariateNormalNoise(loc=pred, scale=torch.ones_like(pred))
    assert torch.allclose(noise_model.sample_to_noise(obs), torch.Tensor([5, 5]))

    # batch size 2
    obs = torch.Tensor([[10, 15], [20, 25]])
    pred = torch.Tensor([[5, 10], [18, 19]])
    noise_model = UnivariateNormalNoise(loc=pred, scale=torch.ones_like(pred))
    assert torch.allclose(noise_model.sample_to_noise(obs), torch.Tensor([[5, 5], [2, 6]]))


def test_noise_to_sample():
    # batch size 1
    noise = torch.Tensor([[5, 5]])
    pred = torch.Tensor([[5, 12]])
    noise_model = UnivariateNormalNoise(loc=pred, scale=torch.ones_like(pred))
    assert torch.allclose(noise_model.noise_to_sample(noise), torch.Tensor([[10, 17]]))

    # no batch size
    noise = torch.Tensor([5, 5])
    pred = torch.Tensor([5, 12])
    noise_model = UnivariateNormalNoise(loc=pred, scale=torch.ones_like(pred))
    assert torch.allclose(noise_model.noise_to_sample(noise), torch.Tensor([10, 17]))
    # batch size 2
    noise = torch.Tensor([[5, 5], [2, 6]])
    pred = torch.Tensor([[5, 12], [18, 19]])
    noise_model = UnivariateNormalNoise(loc=pred, scale=torch.ones_like(pred))
    assert torch.allclose(noise_model.noise_to_sample(noise), torch.Tensor([[10, 17], [20, 25]]))


def test_forward_module():
    dim = 2
    init_log_scale = 1.0
    noise_module = UnivariateNormalNoiseModule(dim, init_log_scale)

    # provide only loc
    loc = torch.randn((dim,))
    noise_model = noise_module(loc)

    samples = torch.randn((dim,))
    assert torch.allclose(noise_model.sample_to_noise(samples), samples - loc)

    # provide loc and log_scale
    log_scale = torch.randn((dim,))
    noise_model = noise_module((loc, log_scale))
    assert torch.allclose(noise_model.sample_to_noise(samples), (samples - loc) / log_scale.exp())
