from typing import Optional

import torch
import torch.distributions as td

from .adjacency_distributions import AdjacencyDistribution


class DeterministicAdjacencyDistribution(AdjacencyDistribution):
    """
    A class that takes an Adjacency Matrix and creates a fixed non-stochastic `distribution`.

    This is useful when we know already know the graph.
    """

    arg_constraints = {"adjacency_matrix": td.constraints.independent(td.constraints.boolean, 1)}

    def __init__(self, adjacency_matrix: torch.Tensor, validate_args: Optional[bool] = None):
        """
        Args:
            adjacency_matrix: The fixed adjacency matrix to be returned by this distribution
            validate_args: Whether to validate the arguments. Passed to the superclass
        """
        num_nodes = adjacency_matrix.shape[-1]

        if validate_args:
            assert len(adjacency_matrix.shape) >= 2, "Logits_exist must be a matrix"
            assert adjacency_matrix.shape == (num_nodes, num_nodes), "Invalid logits_exist shape"

        self.adjacency_matrix = adjacency_matrix  # batch_shape + (num_nodes, num_nodes)

        super().__init__(num_nodes, validate_args=validate_args)

    def relaxed_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0) -> torch.Tensor:
        """
        Sample a binary adjacency matrix from the relaxed distribution.
        For this class this is the same as the sample method.

        Args:
            sample_shape: the shape of the samples to return
            temperature: The temperature of the relaxed distribution
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        _ = temperature
        return self.sample(sample_shape=sample_shape)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample a binary adjacency matrix from the underlying distribution.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        return self.adjacency_matrix.expand(sample_shape + self.adjacency_matrix.shape)

    def entropy(self) -> torch.Tensor:
        """
        Return the entropy of the underlying distribution.

        Returns:
            A tensor of shape (1), with the entropy of the distribution
        """
        return torch.zeros(self.adjacency_matrix.shape[:-2], device=self.adjacency_matrix.device)

    @property
    def mean(self) -> torch.Tensor:
        """
        Return the mean of the underlying distribution.

        This will be the adjacency matrix.

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        return self.adjacency_matrix

    @property
    def mode(self) -> torch.Tensor:
        """
        Return the mode of the underlying distribution.

        This will be the adjacency matrix.

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        return self.adjacency_matrix

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of each tensor from the sample space.

        Raises a NotImplementedError
        """
        raise NotImplementedError("Log probability of a deterministic distribution is not supported.")

    def set_adjacency_matrix(self, matrix: torch.Tensor):
        """Set the adjacency matrix."""
        assert matrix.shape == self.adjacency_matrix.shape, "Matrix must match current shape"
        self.adjacency_matrix = matrix
