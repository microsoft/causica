from typing import Optional

import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import nn
from torch.distributions.utils import logits_to_probs

from causica.distributions.adjacency.adjacency_distributions import AdjacencyDistribution
from causica.distributions.distribution_module import DistributionModule
from causica.distributions.gumbel_binary import gumbel_softmax_binary
from causica.triangular_transformations import fill_triangular


class ENCOAdjacencyDistribution(AdjacencyDistribution):
    """
    The parameterization from the ENCO paper (https://arxiv.org/pdf/2107.10483.pdf).
    For each edge, it parameterizes the existence and orientation separately.

    We represent the orientation logit probabilities (θ) as a vector of length n(n - 1) / 2, (the strictly lower triangle)
    We represent the existence logit probabilities (γ) as a matrix of size (n, n), where the main diagonal is ignored.

    The probability of an edge existing is pᵢⱼ = σ(γᵢⱼ)σ(θᵢⱼ), where:
        σ is the sigmoid function
        θᵢⱼ is the orientation logit and θᵢⱼ = -θⱼᵢ, so the probabilities sum to one.
        γᵢⱼ is the logit the edge exists and is independent for each i, j. Note γᵢⱼ need not equal -γⱼᵢ

    The methods for this class constrct logit(pᵢⱼ), which is then a matrix of independent Bernoulli.
    """

    arg_constraints = {"logits_exist": td.constraints.real, "logits_orient": td.constraints.real}

    def __init__(
        self,
        logits_exist: torch.Tensor,
        logits_orient: torch.Tensor,
        validate_args: Optional[bool] = None,
    ):
        """
        Args:
            logits_exist: The logits for the existence of an edge, a batch_shape + (n, n) tensor
            logits_orient: The logits for the orientation of each edge, a tensor with shape batch_shape + n(n-1)/2
            validate_args: Whether to validate the arguments. Passed to the superclass
        """
        num_nodes = logits_exist.shape[-1]

        if validate_args:
            assert len(logits_exist.shape) >= 2, "Logits_exist must be a matrix, batching is not supported"
            assert logits_exist.shape[-2:] == (num_nodes, num_nodes), "Invalid logits_exist shape"
            assert len(logits_orient.shape) >= 1, "Logits_exist must be 1 dimensional, batching is not supported"
            assert logits_orient.shape[-1] == (num_nodes * (num_nodes - 1)) // 2, "Invalid logits_orient shape"
            assert logits_exist.device == logits_orient.device, "Logits must exist on the same device"

        self.logits_exist = logits_exist
        self.logits_orient = logits_orient

        super().__init__(num_nodes, validate_args=validate_args)

    def _get_independent_bernoulli_logits(self) -> torch.Tensor:
        """
        Construct the matrix logit(pᵢⱼ).


        See the class docstring.
        We use the following derivation
            logit(pᵢⱼ) = - log(pᵢⱼ⁻¹ - 1)
                       = - log((1 + exp(-γᵢⱼ))(1 + exp(-θᵢⱼ)) - 1)
                       = - log(exp(-γᵢⱼ) + exp(-θᵢⱼ) + exp(-θᵢⱼ - γᵢⱼ))
                       = - logsumexp([-γᵢⱼ, -θᵢⱼ, -θᵢⱼ - γᵢⱼ])
        """
        # (..., num_nodes, num_nodes)
        neg_theta = fill_triangular(self.logits_orient, upper=True) - fill_triangular(self.logits_orient, upper=False)
        return -torch.logsumexp(
            torch.stack([-self.logits_exist, neg_theta, neg_theta - self.logits_exist], dim=-1), dim=-1
        )

    @staticmethod
    def base_dist(logits: torch.Tensor) -> td.Distribution:
        """A matrix of independent Bernoulli distributions."""
        return td.Independent(td.Bernoulli(logits=logits), 2)

    def relaxed_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0) -> torch.Tensor:
        """
        Sample from a relaxed distribution. We use a Gumbel Softmax.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        logits = self._get_independent_bernoulli_logits()
        expanded_logits = logits.expand(*(sample_shape + logits.shape))
        samples = gumbel_softmax_binary(logits=expanded_logits, tau=temperature, hard=True)
        return samples * (1.0 - torch.eye(self.num_nodes, device=logits.device))

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample from the underyling independent Bernoulli distribution.

        Gradients will not flow through this method, use relaxed_sample instead.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        logits = self._get_independent_bernoulli_logits()
        samples = self.base_dist(logits).sample(sample_shape=sample_shape)
        torch.diagonal(samples, dim1=-2, dim2=-1).zero_()  # zero the diagonal elements
        return samples

    def entropy(self) -> torch.Tensor:
        """
        Get the entropy of the independent Bernoulli distribution.

        Returns:
            A tensor of shape batch_shape, with the entropy of the distribution
        """
        logits = self._get_independent_bernoulli_logits()
        entropy = F.binary_cross_entropy_with_logits(logits, logits_to_probs(logits, is_binary=True), reduction="none")
        return torch.sum(entropy, dim=(-2, -1)) - torch.sum(torch.diagonal(entropy, dim1=-2, dim2=-1), dim=-1)

    @property
    def mean(self) -> torch.Tensor:
        """
        Return the mean of the underlying independent Bernoulli distribution.

        This will be a matrix with all entries in the interval [0, 1].

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        logits = self._get_independent_bernoulli_logits()
        return self.base_dist(logits).mean * (1.0 - torch.eye(self.num_nodes, device=logits.device))

    @property
    def mode(self) -> torch.Tensor:
        """
        Return the mode of the underlying independent Bernoulli distribution.

        This will be an adjacency matrix.

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        logits = self._get_independent_bernoulli_logits()
        # bernoulli mode can be nan for very small logits, favour sparseness and set to 0
        return torch.nan_to_num(self.base_dist(logits).mode, nan=0.0) * (
            1.0 - torch.eye(self.num_nodes, device=logits.device)
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of each tensor from the sample space

        Args:
            value: a binary matrix of shape value_shape + batch_shape + (n, n)
        Returns:
            A tensor of shape value_shape + batch_shape, with the log probabilities of each tensor in the batch.
        """
        # need to subtract the diagonal log probabilities
        logits = self._get_independent_bernoulli_logits()
        full_log_prob = self.base_dist(logits).log_prob(value)
        diag_log_prob = td.Independent(td.Bernoulli(logits=torch.diagonal(logits, dim1=-2, dim2=-1)), 1).log_prob(
            torch.diagonal(value, dim1=-2, dim2=-1)
        )
        return full_log_prob - diag_log_prob


class ENCOAdjacencyDistributionModule(DistributionModule[ENCOAdjacencyDistribution]):
    """Represents an `ENCOAdjacencyDistributionModule` distribution with learnable logits."""

    def __init__(self, num_nodes: int) -> None:
        super().__init__()
        self.logits_exist = nn.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        self.logits_orient = nn.Parameter(torch.zeros(int(num_nodes * (num_nodes - 1) / 2)), requires_grad=True)

    def forward(self) -> ENCOAdjacencyDistribution:
        return ENCOAdjacencyDistribution(logits_exist=self.logits_exist, logits_orient=self.logits_orient)
