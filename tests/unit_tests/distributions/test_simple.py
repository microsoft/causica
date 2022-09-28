import numpy as np
import pytest
import torch
import torch.nn.functional as F

from causica.distributions.simple import SimpleAdjacencyDistribution


def test_simple_adjacency():
    """Test the simple adjacency distribution"""
    num_nodes = 4
    logits = torch.randn((num_nodes, num_nodes))
    probs = torch.sigmoid(logits)
    dist = SimpleAdjacencyDistribution(logits=logits)
    np.testing.assert_allclose(dist.mean, probs)
    np.testing.assert_allclose(dist.mode, probs.round())
    np.testing.assert_allclose(dist.entropy(), F.binary_cross_entropy_with_logits(logits, probs, reduction="sum"))


@pytest.mark.parametrize("relaxed_sample", [True, False])
def test_simple_log_prob(relaxed_sample: bool):
    """Test the log_prob methods for very low temperatures."""
    num_nodes = 4
    logits = torch.randn((num_nodes, num_nodes))
    dist = SimpleAdjacencyDistribution(logits=logits)
    sample_shape = (3,)
    samples = dist.relaxed_sample(sample_shape, temperature=1e-6) if relaxed_sample else dist.sample(sample_shape)
    np.testing.assert_allclose(
        dist.log_prob(value=samples),
        -torch.sum(
            F.binary_cross_entropy_with_logits(logits.expand(sample_shape + logits.shape), samples, reduction="none"),
            dim=(-1, -2),
        ),
    )
