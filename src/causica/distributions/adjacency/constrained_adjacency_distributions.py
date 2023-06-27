from functools import partial
from typing import Callable, Type

import torch
import torch.distributions as td

from causica.distributions.adjacency.adjacency_distributions import AdjacencyDistribution
from causica.distributions.distribution_module import DistributionModule


class ConstrainedAdjacencyDistribution(AdjacencyDistribution):
    """Adjacency distribution that applies hard constraints to a base distribution.

    Overrides elements produced from the base adjacency distribution with
    - 0 when the corresponding negative_constraint=0
    - 1 when the corresponding positive constraint=1
    - unmodified: all other elements
    """

    arg_constraints = {"positive_constraints": td.constraints.boolean, "negative_constraints": td.constraints.boolean}

    def __init__(
        self, dist: AdjacencyDistribution, positive_constraints: torch.Tensor, negative_constraints: torch.Tensor
    ):
        """
        Args:
            dist (AdjacencyDistribution): Base distribution, event shape matching the postive and negative constraints.
            positive_constraints (torch.Tensor): Positive constraints. 1 means edge is present.
            negative_constraints (torch.Tensor): Negative constraints. 0 means edge is not present.
        """
        if not dist.event_shape == positive_constraints.shape == negative_constraints.shape:
            raise ValueError("The constraints must match the event shape of the distribution.")
        self.dist = dist
        self.positive_constraints = positive_constraints
        self.negative_constraints = negative_constraints

        super().__init__(self.dist.num_nodes)

    def relaxed_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0) -> torch.Tensor:
        """
        Sample a binary adjacency matrix from the relaxed distribution and apply constraints.

        Args:
            sample_shape: the shape of the samples to return
            temperature: The temperature of the relaxed distribution
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        return self._apply_constraints(self.dist.relaxed_sample(sample_shape=sample_shape, temperature=temperature))

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample a binary adjacency matrix from the underlying distribution and apply constraints.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        return self._apply_constraints(self.dist.sample(sample_shape=sample_shape))

    @property
    def mean(self) -> torch.Tensor:
        """
        Return the mean of the underlying distribution and applies the constraints.

        This will be a matrix with all entries in the interval [0, 1].

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        return self._apply_constraints(self.dist.mean)

    @property
    def mode(self) -> torch.Tensor:
        """
        Return the mode of the underlying distribution and applies the constraints.

        This will be an adjacency matrix.

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        return self._apply_constraints(self.dist.mode)

    def entropy(self) -> torch.Tensor:
        """
        Return the entropy of the underlying distribution.

        NOTE: This does not account for the constraints.

        Returns:
            A tensor of shape batch_shape, with the entropy of the distribution
        """
        return self.dist.entropy()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of each tensor from the sample space

        NOTE: This does not account for the constraints.

        Args:
            value: a binary matrix of shape value_shape + batch_shape + (n, n)
        Returns:
            A tensor of shape value_shape + batch_shape, with the log probabilities of each tensor in the batch.
        """
        # entropy might need to be modified to account for constraints
        return self.dist.log_prob(value)

    def _apply_constraints(self, G: torch.Tensor) -> torch.Tensor:
        """Return G with the positive and negative constraints applied."""
        return 1.0 - (1.0 - G * self.negative_constraints) * (~self.positive_constraints)


def get_graph_constraint(graph_constraint_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Converts graph constraint matrix into a positive and negative matrix for easier usage.

    Args:
        graph_constraint_matrix: Graph constraints: 0 = no edge, 1 = edge, nan: no constraint. Should be a square matrix
                                 with the lengths matching the number of nodes in the graph.

    Returns:
        A tuple of (positive_constraints, negative_constraint). See ConstrainedAdjacencyDistribution for their
        interpretation.
    """
    assert graph_constraint_matrix.ndim == 2, "Constraint matrix must be 2D."
    assert graph_constraint_matrix.shape[0] == graph_constraint_matrix.shape[1], "Constraint matrix must be square."
    # Mask self-edges
    mask = ~torch.eye(graph_constraint_matrix.shape[0], dtype=torch.bool, device=graph_constraint_matrix.device)

    positive_constraints = mask * torch.nan_to_num(graph_constraint_matrix, nan=0).to(
        dtype=torch.bool, non_blocking=True
    )
    negative_constraints = torch.nan_to_num(graph_constraint_matrix, nan=1).to(dtype=torch.bool, non_blocking=True)
    return positive_constraints, negative_constraints


def _create_distribution(
    dist_class: Type[AdjacencyDistribution], *args, graph_constraint_matrix: torch.Tensor, **kwargs
) -> ConstrainedAdjacencyDistribution:
    """Utility function for generating a constrained adjacency distribution with a base distribution.

    Args:
        dist_class (Type[AdjacencyDistribution]): Type of the base adjacency distribution.
        graph_constraint_matrix: Graph constraints: 0 = no edge, 1 = edge, nan: no constraint. Must match the event
                                 shape of the distribution.

    Returns:
        ConstrainedAdjacencyDistribution: Constrained adjacency distribution.
    """

    positive_constraints, negative_constraints = get_graph_constraint(graph_constraint_matrix)

    dist = dist_class(*args, **kwargs)
    return ConstrainedAdjacencyDistribution(
        dist, positive_constraints=positive_constraints, negative_constraints=negative_constraints
    )


def constrained_adjacency(
    dist_class: Type[AdjacencyDistribution],
) -> Callable[..., ConstrainedAdjacencyDistribution]:
    """Utility function that returns a function constructing a constrained adjacency distribution.

    Args:
        dist_class: Type of the base adjacency distribution.
        graph_constraint_matrix: Graph constraints: 0 = no edge, 1 = edge, nan: no constraint. Must match the event
                                 shape of the distribution.

    Returns:
        Callable[..., ConstrainedAdjacencyDistribution]: Utility function creating a ConstrainedAdjacencyDistribution.
    """

    return partial(
        _create_distribution,
        dist_class=dist_class,
    )


class ConstrainedAdjacency(DistributionModule[AdjacencyDistribution]):
    """A constrained adjacency distribution module where certain parts edges of in the adjacency matrix are locked."""

    def __init__(
        self, adjacency_distribution: DistributionModule[AdjacencyDistribution], graph_constraint_matrix: torch.Tensor
    ):
        """
        Args:
            adjacency_distribution: Underlying adjacency distribution module.
            graph_constraint_matrix: Constraint matrix with edges defined according to `get_graph_constraint`.
        """
        super().__init__()
        self.adjacency_distribution = adjacency_distribution
        positive_constraints, negative_constraints = get_graph_constraint(graph_constraint_matrix)
        self.positive_constraints: torch.Tensor
        self.negative_constraints: torch.Tensor
        self.register_buffer("positive_constraints", positive_constraints)
        self.register_buffer("negative_constraints", negative_constraints)

    def forward(self) -> ConstrainedAdjacencyDistribution:
        return ConstrainedAdjacencyDistribution(
            self.adjacency_distribution(),
            positive_constraints=self.positive_constraints,
            negative_constraints=self.negative_constraints,
        )
