from typing import Type

import pytest
import torch

from causica.distributions.adjacency.adjacency_distributions import AdjacencyDistribution
from causica.distributions.adjacency.constrained_adjacency_distributions import constrained_adjacency
from causica.distributions.adjacency.enco import ENCOAdjacencyDistribution
from causica.distributions.adjacency.temporal_adjacency_distributions import (
    RhinoLaggedAdjacencyDistribution,
    TemporalAdjacencyDistribution,
)
from causica.distributions.adjacency.three_way import ThreeWayAdjacencyDistribution

DIST_CLASSES = [
    ENCOAdjacencyDistribution,
    ThreeWayAdjacencyDistribution,
]

TEMPORAL_DIST_CLASSES = [TemporalAdjacencyDistribution]


@pytest.mark.parametrize("dist_class", DIST_CLASSES)
@pytest.mark.parametrize("num_nodes", [3, 100, 500])
@pytest.mark.parametrize("force_edge_from_diagonal", [1, 2, 3])
@pytest.mark.parametrize("exist_force_edge", [True, False])
def test_constrained_adjacency(
    dist_class: Type[AdjacencyDistribution],
    num_nodes: int,
    force_edge_from_diagonal: int,
    exist_force_edge: bool,
):
    """Tests that a constrained distribution can be set up that forces no edges from the given diagonal.

    Args:
        dist_class: The class of the adjacency distribution to test.
        num_nodes: Number of nodes represented by the adjacency matrix.
        force_edge_from_diagonal: Specifies which diagonal of the adjacency matrix to force to 1.
    """

    ones = torch.ones((num_nodes, num_nodes), dtype=torch.bool) * exist_force_edge
    nans = torch.full((num_nodes, num_nodes), fill_value=torch.nan)
    if force_edge_from_diagonal == 0:
        ones = ones & ~torch.eye(num_nodes, dtype=torch.bool)

    # Set all edges but one diagonal to nan
    constraints = (
        ones
        + torch.tril(nans, diagonal=force_edge_from_diagonal - 1)
        + torch.triu(nans, diagonal=force_edge_from_diagonal + 1)
    )

    mask = ~torch.isnan(constraints)

    constrained_dist_cls = constrained_adjacency(dist_class)
    if dist_class is ThreeWayAdjacencyDistribution:
        logits = torch.nn.Parameter(torch.zeros(((num_nodes * (num_nodes - 1)) // 2, 3)))
        constrained_dist = constrained_dist_cls(logits=logits, graph_constraint_matrix=constraints)
    elif dist_class is ENCOAdjacencyDistribution:
        logits_exist = torch.nn.Parameter(torch.zeros((num_nodes, num_nodes)))
        logits_orient = torch.nn.Parameter(torch.zeros(((num_nodes * (num_nodes - 1)) // 2,)))
        constrained_dist = constrained_dist_cls(
            logits_exist=logits_exist, logits_orient=logits_orient, graph_constraint_matrix=constraints
        )
    else:
        raise ValueError("Unrecognised Class")

    assert torch.allclose(constrained_dist.sample()[mask], constraints[mask])


@pytest.mark.parametrize("dist_class", TEMPORAL_DIST_CLASSES)
@pytest.mark.parametrize("num_nodes", [3, 100, 500])
@pytest.mark.parametrize("force_edge_from_diagonal", [1, 2, 3])
@pytest.mark.parametrize("exist_force_edge", [True, False])
@pytest.mark.parametrize("context_length", [1, 3])
def test_temporal_constrained_adjacency(
    dist_class: Type[TemporalAdjacencyDistribution],
    num_nodes: int,
    force_edge_from_diagonal: int,
    exist_force_edge: bool,
    context_length: int,
):
    """Tests that a temporal constrained distribution can be set up that forces no edges from the given diagonal.

    Args:
        dist_class: The class of the temporal adjacency distribution to test.
        num_nodes: Number of nodes represented by the adjacency matrix.
        force_edge_from_diagonal: Specifies which diagonal of the adjacency matrix to force to 1.
    """

    assert context_length >= 1
    ones = torch.ones((context_length, num_nodes, num_nodes), dtype=torch.bool) * exist_force_edge
    nans = torch.full((context_length, num_nodes, num_nodes), fill_value=torch.nan)
    if force_edge_from_diagonal == 0:
        ones[-1, :, :] = ones[-1, :, :] & ~torch.eye(num_nodes, dtype=torch.bool)
    # Set all edges but one diagonal to nan
    constraints = (
        ones
        + torch.tril(nans, diagonal=force_edge_from_diagonal - 1)
        + torch.triu(nans, diagonal=force_edge_from_diagonal + 1)
    )

    mask = ~torch.isnan(constraints)

    constrained_dist_cls = constrained_adjacency(dist_class)

    logits_exist = torch.nn.Parameter(torch.zeros((context_length, num_nodes, num_nodes)))
    logits_orient = torch.nn.Parameter(torch.zeros(((num_nodes * (num_nodes - 1)) // 2,)))
    lagged_dist = (
        RhinoLaggedAdjacencyDistribution(logits_edge=logits_exist[..., :-1, :, :], lags=context_length - 1)
        if context_length > 1
        else None
    )
    inst_dist = ENCOAdjacencyDistribution(logits_exist=logits_exist[..., -1, :, :], logits_orient=logits_orient)
    constrained_dist = constrained_dist_cls(
        instantaneous_distribution=inst_dist, lagged_distribution=lagged_dist, graph_constraint_matrix=constraints
    )

    assert torch.allclose(constrained_dist.sample()[mask], constraints[mask])
