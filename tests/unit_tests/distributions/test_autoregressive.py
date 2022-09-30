import numpy as np
import pytest
import torch

from causica.distributions.autoregressive import AutoregressiveDistribution
from causica.utils.nri_utils import enum_all_graphs


@pytest.mark.parametrize("num_nodes", [2, 3, 4])
def test_autoregressive(num_nodes: int):
    """Test Autoregressive by passing all the graphs and checking that their probs sum to 1."""
    all_graphs = torch.from_numpy(enum_all_graphs(num_nodes=num_nodes, dags_only=False).astype(np.float32))
    dist = AutoregressiveDistribution(num_nodes=num_nodes)
    np.testing.assert_allclose(torch.logsumexp(dist.log_prob(all_graphs), dim=0).item(), 0.0, atol=1e-06)
