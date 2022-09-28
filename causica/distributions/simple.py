from typing import Optional

import torch
import torch.distributions as td

from .adjacency_distributions import AdjacencyDistribution
from .utils import gumbel_softmax_binary


class SimpleAdjacencyDistribution(AdjacencyDistribution):
    """
    A class that takes a matrix of logits and creates a matrix of independent Bernoulli distributions
    """

    arg_constraints = {"logits": td.constraints.real}

    def __init__(self, logits: torch.Tensor, validate_args: Optional[bool] = None):
        """
        Args:
            logits: The logits for the existence of an edge, a (n, n) matrix
            validate_args: Whether to validate the arguments. Passed to the superclass
        """
        num_nodes = logits.shape[-1]

        if validate_args:
            assert len(logits.shape) >= 2, "Logits_exist must be a matrix, batching is not supported"
            assert logits.shape[-2:] == (num_nodes, num_nodes), "Invalid logits_exist shape"

        self.logits = logits  # (..., num_nodes, num_nodes)

        super().__init__(num_nodes, validate_args=validate_args)

    @staticmethod
    def base_dist(logits: torch.Tensor) -> td.Distribution:
        """A matrix of independent Bernoulli distributions."""
        return td.Independent(td.Bernoulli(logits=logits), 2)

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
        shape = sample_shape + self.logits.shape
        return gumbel_softmax_binary(logits=self.logits.expand(*shape), tau=temperature, hard=True)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample a binary adjacency matrix from the underlying distribution.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        return self.base_dist(self.logits).sample(sample_shape=sample_shape)

    def entropy(self) -> torch.Tensor:
        """
        Return the entropy of the underlying distribution.

        Returns:
            A tensor of shape batch_shape, with the entropy of the distribution
        """
        return self.base_dist(self.logits).entropy()

    @property
    def mean(self) -> torch.Tensor:
        """
        Return the mean of the underlying distribution.

        This will be the adjacency matrix.

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        return self.base_dist(self.logits).mean

    @property
    def mode(self) -> torch.Tensor:
        """
        Return the mode of the underlying distribution.

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        return self.base_dist(self.logits).mode

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of each tensor from the sample space.

        Args:
            value: a binary matrix of shape value_shape + batch_shape + (n, n)
        Returns:
            A tensor of shape value_shape + batch_shape, with the log probabilities of each tensor in the batch.
        """
        return self.base_dist(self.logits).log_prob(value)
