import pytest
import torch

from causica.distributions.adjacency.constrained_adjacency_distributions import constrained_adjacency
from causica.distributions.adjacency.three_way import ThreeWayAdjacencyDistribution


@pytest.mark.parametrize("num_nodes", [3, 100, 500])
@pytest.mark.parametrize("force_edge_from_diagonal", [1, 2, 3])
def test_constrained_adjacency(num_nodes: int, force_edge_from_diagonal: int):
    """Tests that a constrained distribution can be set up that forces no edges from the given diagonal.

    Args:
        num_nodes: Number of nodes represented by the adjacency matrix.
        force_edge_from_diagonal: Specifies which diagonal of the adjacency matrix to force to 1.
    """
    nans = torch.full((num_nodes, num_nodes), torch.nan, dtype=torch.bool)
    constraints = torch.ones_like(nans) + torch.tril(nans, diagonal=force_edge_from_diagonal)
    constrained_dist_cls = constrained_adjacency(ThreeWayAdjacencyDistribution, constraints)
    logits = torch.nn.Parameter(torch.zeros(((num_nodes * (num_nodes - 1)) // 2, 3)))
    constrained_dist = constrained_dist_cls(logits=logits)
    assert torch.equal(
        torch.triu(constrained_dist.sample(), diagonal=force_edge_from_diagonal),
        torch.triu(constraints, diagonal=force_edge_from_diagonal),
    )
