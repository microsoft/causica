import numpy as np
import torch
from torch.distributions.utils import probs_to_logits

from causica.distributions import ThreeWayAdjacencyDistribution


def test_threeway_entropy():
    """Test entropy is correct for a known distribution"""
    num_nodes = 3
    logits = torch.nn.Parameter(
        torch.zeros(((num_nodes * (num_nodes - 1)) // 2, 3))
    )  # create a max entropy distribution
    dist = ThreeWayAdjacencyDistribution(logits=logits)
    entropy = dist.entropy()
    np.testing.assert_allclose(entropy.detach().numpy(), logits.shape[0] * np.log(3))
    # check the gradient of the max entropy is ~zero
    entropy.backward()
    np.testing.assert_allclose(logits.grad, np.zeros_like(logits.detach().numpy()), atol=1e-7)


def test_threeway():
    """Test ThreeWay methods are correct for a known distribution."""
    probs = torch.Tensor([[0.4, 0.25, 0.35]])
    dist = ThreeWayAdjacencyDistribution(logits=probs_to_logits(probs))
    np.testing.assert_allclose(dist.mean, np.array([[0.0, 0.25], [0.4, 0.0]]), rtol=1e-6)
    np.testing.assert_allclose(dist.mode, np.array([[0.0, 0.0], [1.0, 0.0]]), rtol=1e-6)
    samples = torch.tensor([[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
    np.testing.assert_allclose(dist.log_prob(samples), np.array([np.log(0.25), np.log(0.4), np.log(0.35)]))
