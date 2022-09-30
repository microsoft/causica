import numpy as np
import pytest
import torch

from causica.distributions.deterministic import DeterministicAdjacencyDistribution


def test_deterministic():
    """Test the deterministic distribution."""
    num_nodes = 4
    adj = torch.randint(0, 2, (num_nodes, num_nodes))
    torch.diagonal(adj).zero_()
    dist = DeterministicAdjacencyDistribution(adjacency_matrix=adj)
    np.testing.assert_allclose(dist.mean, adj)
    np.testing.assert_allclose(dist.mode, adj)
    assert (dist.sample((4, 4)) == adj[None, None, ...]).all()
    assert (dist.relaxed_sample((4, 4)) == adj[None, None, ...]).all()
    np.testing.assert_allclose(dist.entropy(), torch.zeros((1,)))
    with pytest.raises(NotImplementedError):
        dist.log_prob((2,))
