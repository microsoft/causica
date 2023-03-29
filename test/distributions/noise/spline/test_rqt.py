import numpy as np
import pytest
import torch

from causica.distributions.noise.spline.rational_quadratic_transform import normalize_bins

SHAPES = [(5,), (2, 3), (3, 2)]
TAIL_BOUND = 0.5


@pytest.mark.parametrize("shape", SHAPES)
def test_normalize_bins(shape):
    bins = torch.ones(shape)
    normalized_bins, cum_normalized_bins = normalize_bins(bins, tail_bound=TAIL_BOUND)
    assert normalized_bins.shape == bins.shape
    assert cum_normalized_bins.shape[:-1] == bins.shape[:-1]
    assert cum_normalized_bins.shape[-1] == bins.shape[-1] + 1
    np.testing.assert_allclose(normalized_bins.numpy(), 2 * TAIL_BOUND / shape[-1])
    np.testing.assert_allclose(cum_normalized_bins[..., 0], -TAIL_BOUND)
    np.testing.assert_allclose(cum_normalized_bins[..., -1], TAIL_BOUND, rtol=1e-6)
