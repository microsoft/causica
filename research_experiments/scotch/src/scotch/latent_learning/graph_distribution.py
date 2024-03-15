from typing import Optional

import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import nn
from torch.distributions.utils import logits_to_probs

from causica.distributions.adjacency.adjacency_distributions import AdjacencyDistribution
from causica.distributions.distribution_module import DistributionModule
from causica.distributions.gumbel_binary import gumbel_softmax_binary


class BernoulliDigraphDistribution(AdjacencyDistribution):
    """Directed graph distribution, with independent Bernoulli distributions for each edge. Allows for self-loops.

    The probability of an edge existing is pᵢⱼ = σ(γᵢⱼ). where γᵢⱼ is the logit the edge exists and is independent for
    each i, j.
    """

    arg_constraints = {}

    def __init__(self, logits: torch.Tensor, validate_args: Optional[bool] = None):
        num_nodes = logits.shape[-1]
        super().__init__(num_nodes, validate_args=validate_args)

        self.logits = logits

    @staticmethod
    def base_dist(logits: torch.Tensor) -> td.Distribution:
        """A matrix of independent Bernoulli distributions.

        Args:
            logits: The logits for the existence of an edge; shape == batch_shape + (num_nodes, num_nodes)
        """
        return td.Independent(td.Bernoulli(logits=logits), 2)

    def relaxed_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0) -> torch.Tensor:
        """
        Sample from a relaxed distribution. We use a Gumbel Softmax.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        expanded_logits = self.logits.expand(*(sample_shape + self.logits.shape))
        return gumbel_softmax_binary(logits=expanded_logits, tau=temperature, hard=True)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample from the underyling independent Bernoulli distribution.

        Gradients will not flow through this method, use relaxed_sample instead.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        return self.base_dist(self.logits).sample(sample_shape=sample_shape)

    def entropy(self) -> torch.Tensor:
        """
        Get the entropy of the independent Bernoulli distribution.

        Returns:
            A tensor of shape batch_shape, with the entropy of the distribution
        """
        entropy = F.binary_cross_entropy_with_logits(
            self.logits, logits_to_probs(self.logits, is_binary=True), reduction="none"
        )
        return torch.sum(entropy, dim=(-2, -1))

    @property
    def mean(self) -> torch.Tensor:
        """
        Return the mean of the underlying independent Bernoulli distribution.

        This will be a matrix with all entries in the interval [0, 1].

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        return self.base_dist(self.logits).mean

    @property
    def mode(self) -> torch.Tensor:
        """
        Return the mode of the underlying independent Bernoulli distribution.

        This will be an adjacency matrix.

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        # bernoulli mode can be nan for very small logits, favour sparseness and set to 0
        return torch.nan_to_num(self.base_dist(self.logits).mode, nan=0.0)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of each tensor from the sample space

        Args:
            value: a binary matrix of shape value_shape + batch_shape + (num_nodes, num_nodes)
        Returns:
            A tensor of shape value_shape + batch_shape, with the log probabilities of each tensor in the batch.
        """
        return self.base_dist(self.logits).log_prob(value)


class BernoulliDigraphDistributionModule(DistributionModule[BernoulliDigraphDistribution]):
    """Represents an `BernoulliDigraphDistribution` distribution with learnable logits."""

    def __init__(self, num_nodes: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)  # prob 0.5

    def forward(self) -> BernoulliDigraphDistribution:
        return BernoulliDigraphDistribution(logits=self.logits)
