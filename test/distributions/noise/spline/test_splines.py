"""
Spline Tests - Note these only compare the behaviour to the old behaviour.
TODO(JJ) update these tests to be standalone
"""
from functools import partial

import torch

from causica.distributions.noise.spline import SplineNoiseModule

INPUT_DIM = 4

torch.manual_seed(123)

assert_close = partial(torch.testing.assert_close, rtol=2e-6, atol=2e-5)


def test_sample_to_noise():
    """Test sample to noise and noise to sample work as expected."""
    module = SplineNoiseModule(dim=INPUT_DIM, flow_steps=1)
    output_bias = torch.randn(INPUT_DIM)
    dist = module(output_bias)
    samples = dist.sample((100,))
    noise = dist.sample_to_noise(samples)
    recon_samples = dist.noise_to_sample(noise)
    torch.testing.assert_close(samples, recon_samples)
    torch.testing.assert_close(noise, dist.sample_to_noise(recon_samples))
