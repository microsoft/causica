import pytest
import torch

from causica.distributions.adjacency.constrained_adjacency_distributions import constrained_adjacency
from causica.distributions.adjacency.three_way import ThreeWayAdjacencyDistribution


@pytest.mark.parametrize("num_nodes", [3, 100, 500])
@pytest.mark.parametrize("force_edge_from_diagonal", [1, 2, 3])
@pytest.mark.parametrize("exist_force_edge", [True, False])
def test_constrained_adjacency(num_nodes: int, force_edge_from_diagonal: int, exist_force_edge: bool):
    """Tests that a constrained distribution can be set up that forces no edges from the given diagonal.

    Args:
        num_nodes: Number of nodes represented by the adjacency matrix.
        force_edge_from_diagonal: Specifies which diagonal of the adjacency matrix to force to 1.
    """

    ones = torch.ones((num_nodes, num_nodes), dtype=torch.bool) * exist_force_edge
    nans = torch.full((num_nodes, num_nodes), fill_value=torch.nan)

    # Set all edges but one diagonal to nan
    constraints = (
        ones
        + torch.tril(nans, diagonal=force_edge_from_diagonal - 1)
        + torch.triu(nans, diagonal=force_edge_from_diagonal + 1)
    )

    mask = ~torch.isnan(constraints)

    constrained_dist_cls = constrained_adjacency(ThreeWayAdjacencyDistribution)
    logits = torch.nn.Parameter(torch.zeros(((num_nodes * (num_nodes - 1)) // 2, 3)))
    constrained_dist = constrained_dist_cls(logits=logits, graph_constraint_matrix=constraints)

    assert torch.allclose(constrained_dist.sample()[mask], constraints[mask])
