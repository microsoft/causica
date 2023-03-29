import math

import pytest
import torch
import torch.distributions as td

from causica.distributions.noise import BernoulliNoise


def sample_logistic_noise(n_samples, input_dim, base_logits):
    """
    Samples a Logistic random variable that can be used to sample this variable.
    This method does **not** return hard-thresholded samples.
    Args:
        n_samples

    Returns:
        samples (Nsamples, input_dim)
    """
    # The difference of two independent Gumbel(0, 1) variables is a Logistic random variable
    dist = td.Gumbel(torch.tensor(0.0), torch.tensor(1.0))
    g0 = dist.sample((n_samples, input_dim))
    g1 = dist.sample((n_samples, input_dim))
    return g1 - g0 + base_logits


def test_init():
    base_logits = torch.Tensor([0.3, 0.2, 0.1])

    # no batch
    x_hat = torch.Tensor([0.3, 0.2, 0.1])
    noise_model = BernoulliNoise(x_hat, base_logits)

    assert noise_model.logits.shape == torch.Size([3])
    assert torch.equal(noise_model.logits, torch.Tensor([0.6, 0.4, 0.2]))

    # batch size 1
    x_hat = torch.Tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]])
    noise_model = BernoulliNoise(x_hat, base_logits)

    assert noise_model.logits.shape == torch.Size([2, 3])
    torch.testing.assert_close(noise_model.logits, torch.Tensor([[0.6, 0.4, 0.2], [0.7, 0.7, 0.7]]))


@pytest.mark.parametrize(
    "base_logits,x_hat",
    [
        (0.0, torch.tensor([0.0, 0.0, 0.0])),
        (1.0, torch.tensor([0.0, 0.0, 0.0])),
        (0.0, torch.tensor([0.1, 0.5, 1.0])),
        (1.0, torch.tensor([1.0, 2.0, 3.0])),
        (1.0, torch.tensor([-1.0, 2.0, -1.0])),
    ],
)
def test_sample_to_noise(x_hat, base_logits):
    n = 5000
    x_hat = x_hat.repeat(n, 1)
    noise_model = BernoulliNoise(delta_logits=x_hat, base_logits=base_logits)
    logistic_noise = sample_logistic_noise(n, x_hat.shape[-1], base_logits)  # [5000,3], 0,1
    # generate samples

    samples = noise_model.noise_to_sample(logistic_noise)  # [5000, 3]
    # get posterior noise
    post_noise = noise_model.sample_to_noise(samples)
    # This is 8 sigma of a Logistic variable
    eight_sigma = 8 * math.pi / math.sqrt(3 * n)
    assert abs(logistic_noise.mean() - post_noise.mean()) < eight_sigma

    # get reconstruct samples
    r_samples = noise_model.noise_to_sample(post_noise)
    assert torch.allclose(samples, r_samples)
