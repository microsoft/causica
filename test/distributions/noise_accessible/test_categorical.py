import math

import pytest
import torch

from causica.distributions import NoiseAccessibleCategorical


def test_init():
    base_logits = torch.Tensor([0.3, 0.2, 0.1])

    # no batch
    x_hat = torch.Tensor([0.3, 0.2, 0.1])
    noise_model = NoiseAccessibleCategorical(x_hat, base_logits)

    assert noise_model.logits.shape == torch.Size([3])
    logits = base_logits + x_hat
    assert torch.equal(noise_model.logits, logits - logits.logsumexp(dim=-1, keepdim=True))

    # batch size 1
    x_hat = torch.Tensor([[0.3, 0.2, 0.1], [0.4, 0.5, 0.6]])
    noise_model = NoiseAccessibleCategorical(x_hat, base_logits)

    assert noise_model.logits.shape == torch.Size([2, 3])
    logits = base_logits + x_hat
    assert torch.equal(noise_model.logits, logits - logits.logsumexp(dim=-1, keepdim=True))


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
def test_noise_reconstruction(base_logits, x_hat):
    n = 5000
    x_hat = x_hat.repeat(n, 1)
    noise_model = NoiseAccessibleCategorical(delta_logits=x_hat, base_logits=base_logits)
    # generate samples
    samples = noise_model.sample()  # [5000, 3]

    # test samples
    noise = noise_model.sample_to_noise(samples)
    # get posterior sample again
    post_sample = noise_model.noise_to_sample(noise)
    assert torch.allclose(samples, post_sample)

    # test noise
    post_noise = noise_model.sample_to_noise(post_sample)
    # This is 8 sigma of a Logistic variable
    eight_sigma = 8 * math.pi / math.sqrt(3 * n)
    assert abs(noise.mean() - post_noise.mean()) < eight_sigma
