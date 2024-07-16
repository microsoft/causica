import abc
from typing import Optional

import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import nn
from torch.distributions.utils import logits_to_probs

from causica.distributions.adjacency.adjacency_distributions import AdjacencyDistribution
from causica.distributions.distribution_module import DistributionModule
from causica.distributions.gumbel_binary import gumbel_softmax_binary


class LaggedAdjacencyDistribution(AdjacencyDistribution, abc.ABC):
    """
    Probability distributions of the lagged adjacency matrix for causal timeseries model.

    The main functionality is similar to `AdjacencyDistribution`. The main differences are that we additional have
    `lag` parameters to indicate the window size of the temporal adjacency matrix. E.g. lag=3
    indicates we only consider previous 3 step of observations. `lags` equals to `context_length` - 1.

    Another difference is that we do not disable the diagonals of the lagged adjacency matries and no dagness constraints as well.

    One single sample from this lagged distribution should have shape (lags, num_nodes, num_nodes). lags must be greater than 0.
    For multple samples, the shape will be sample_shape + batch_shape + (lags, num_nodes, num_nodes)
    """

    def __init__(self, num_nodes: int, lags: int, validate_args: Optional[bool] = None):
        assert lags > 0, "Number of lags must be greater than 0"
        self.lags = lags
        assert num_nodes > 0, "Number of nodes in the graph must be greater than 0"
        self.num_nodes = num_nodes
        event_shape = torch.Size((lags, num_nodes, num_nodes))

        super(AdjacencyDistribution, self).__init__(event_shape=event_shape, validate_args=validate_args)


class TemporalAdjacencyDistribution(td.Distribution):
    """Probability distributions of the temporal adjacency matrix for causal timeseries model.

    Combines an instantaneous AdjacencyDistribution and temporal LaggedAdjacencyDistribution to form a proper temporal adjacency matrix.

    One single sample from this distribution should have shape (context_length, num_nodes, num_nodes)
    For multple samples, the shape will be sample_shape + batch_shape + (context_length, num_nodes, num_nodes)

    For single sample, graph[-1] will be the instantaneous graph and graph[0] will be the last lagged graph (if context_length > 1).
    """

    support = td.constraints.independent(td.constraints.boolean, 1)
    arg_constraints = {}

    def __init__(
        self,
        instantaneous_distribution: AdjacencyDistribution,
        lagged_distribution: Optional[LaggedAdjacencyDistribution],
        validate_args: Optional[bool] = None,
    ):
        """
        Args:
            instantaneous_distribution: The distribution of the instantaneous adjacency matrix
            lagged_distribution: The distribution of the lagged adjacency matrix
            validate_args: Whether to validate the input arguments
        """
        if lagged_distribution is not None:
            self.lags = lagged_distribution.lags
            assert (
                instantaneous_distribution.num_nodes == lagged_distribution.num_nodes
            ), "Number of nodes in the graph must be the same"
        else:
            self.lags = 0
        self.context_length = self.lags + 1
        self.num_nodes = instantaneous_distribution.num_nodes
        event_shape = torch.Size((self.context_length, self.num_nodes, self.num_nodes))
        self.inst_dist = instantaneous_distribution
        self.lagged_dist = lagged_distribution

        super().__init__(event_shape=event_shape, validate_args=validate_args)

    def relaxed_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0) -> torch.Tensor:
        """Sample a binary temporal adjacency matrix from the relaxed distribution.

        This relies on the relaxed sampling method of both the instantaneous and lagged distributions, so that gradients
        can flow through. To obtain hard samples that stops gradients, use the `sample` method.

        Args:
            sample_shape: the shape of the samples to return
            temperature: The temperature of the relaxed distribution

        Returns:
            A tensor of shape sample_shape + batch_shape + (context_length, num_nodes, num_nodes)
        """
        # sample inst adj matrix with shape sample_shape + batch_shape + (num_nodes, num_nodes)
        inst_adj_matrix = self.inst_dist.relaxed_sample(sample_shape, temperature)
        if self.lagged_dist is None:
            return inst_adj_matrix.unsqueeze(-3)  # sample_shape + batch_shape + (1, num_nodes, num_nodes)
        # sample lagged adj matrix with shape sample_shape + batch_shape + (lags, num_nodes, num_nodes)
        lagged_adj_matrix = self.lagged_dist.relaxed_sample(sample_shape, temperature)

        return torch.cat(
            [lagged_adj_matrix, inst_adj_matrix.unsqueeze(-3)], dim=-3
        )  # sample_shape + batch_shape + (context_length, num_nodes, num_nodes)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample a binary temporal adjacency matrix from the underlying distribution.

        The gradients will not flow through this method, use `relaxed_sample` instead.

        Args:
            sample_shape: the shape of the samples to return

        Returns:
            A tensor of shape sample_shape + batch_shape + (context_length, num_nodes, num_nodes)
        """
        inst_adj_matrix = self.inst_dist.sample(sample_shape)  # sample_shape + batch_shape + (num_nodes, num_nodes)
        if self.lagged_dist is None:
            return inst_adj_matrix.unsqueeze(-3)  # sample_shape + batch_shape + (1, num_nodes, num_nodes)

        lagged_adj_matrix = self.lagged_dist.sample(
            sample_shape
        )  # sample_shape + batch_shape + (lags, num_nodes, num_nodes)
        return torch.cat(
            [lagged_adj_matrix, inst_adj_matrix.unsqueeze(-3)], dim=-3
        )  # sample_shape + batch_shape + (context_length, num_nodes, num_nodes)

    def entropy(self) -> torch.Tensor:
        """
        Return the entropy of the underlying temporal adjacency distribution.

        Returns:
            A tensor of shape batch_shape, with the entropy of the distribution
        """
        if self.lagged_dist is None:
            return self.inst_dist.entropy()
        return self.inst_dist.entropy() + self.lagged_dist.entropy()

    @property
    def mean(self) -> torch.Tensor:
        """
        Return the mean of the underlying temporal adjaccency distribution.

        This will be a matrix with all entries in the interval [0, 1].

        Returns:
            A tensor of shape batch_shape + (context_length, num_nodes, num_nodes)
        """
        if self.lagged_dist is None:
            return self.inst_dist.mean.unsqueeze(-3)  # batch_shape + (1, num_nodes, num_nodes)

        return torch.cat(
            [self.lagged_dist.mean, self.inst_dist.mean.unsqueeze(-3)], dim=-3
        )  # batch_shape + (context_length, num_nodes, num_nodes)

    @property
    def mode(self) -> torch.Tensor:
        """
        Return the mode of the underlying temporal adjacency distribution.

        This will be an temporal adjacency matrix.

        Returns:
            A tensor of shape batch_shape + (context_length, num_nodes, num_nodes)
        """
        if self.lagged_dist is None:
            return self.inst_dist.mode.unsqueeze(-3)  # batch_shape + (1, num_nodes, num_nodes)

        return torch.cat(
            [self.lagged_dist.mode, self.inst_dist.mode.unsqueeze(-3)], dim=-3
        )  # batch_shape + (context_length, num_nodes, num_nodes)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of each tensor from the sample space

        Args:
            value: a binary matrix of shape value_shape + batch_shape + (context_length, n, n)
        Returns:
            A tensor of shape value_shape + batch_shape, with the log probabilities of each tensor in the batch.
        """
        inst_adj_matrix = value[..., -1, :, :]  # value_shape + batch_shape + (num_nodes, num_nodes)
        if self.lagged_dist is None:
            return self.inst_dist.log_prob(inst_adj_matrix)  # value_shape + batch_shape

        lagged_adj_matrix = value[..., :-1, :, :]  # value_shape + batch_shape + (lags, num_nodes, num_nodes)
        return self.inst_dist.log_prob(inst_adj_matrix) + self.lagged_dist.log_prob(lagged_adj_matrix)


class RhinoLaggedAdjacencyDistribution(LaggedAdjacencyDistribution):
    """
    This implements the adjacency distribution for lagged adj matrix.
    """

    arg_constraints = {"logits_edge": td.constraints.real}

    def __init__(
        self,
        logits_edge: torch.Tensor,
        lags: int,
        validate_args: Optional[bool] = None,
    ):
        """
        Args:
            logits_edge: The logits for the edge existence. The shape is a batch_shape + (lags, n, n) tensor.
            lags: The number of lags in the temporal adjacency matrix. It must be positive.
            validate_args: Whether to validate the arguments. Passed to the superclass
        """
        assert lags == logits_edge.shape[-3], "lags must match the number of lags in logits_edge"
        assert lags > 0, "lags must be a positive integer"
        num_nodes = logits_edge.shape[-1]

        if validate_args:
            assert len(logits_edge.shape) >= 3, "logits_exist must be a 3D tensor with shape (lags, n, n)"
            assert logits_edge.shape[-3:] == (lags, num_nodes, num_nodes), "Invalid logits_edge shape"

        self.logits_edge = logits_edge
        self.lags = lags

        super().__init__(num_nodes, lags, validate_args=validate_args)

    def _get_independent_bernoulli_logits(self) -> torch.Tensor:
        """
        Construct the lagged matrix logit(pᵢⱼ).
        The output will have shape [..., lags, node, node]

        """
        # Lagged adjacency logits
        return self.logits_edge  # [..., lags, node, node]

    @staticmethod
    def base_dist(logits: torch.Tensor) -> td.Distribution:
        """A matrix of independent Bernoulli distributions."""
        return td.Independent(td.Bernoulli(logits=logits), 3)

    def relaxed_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0) -> torch.Tensor:
        """
        Sample from a relaxed distribution. We use a Gumbel Softmax.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (lags, num_nodes, num_nodes)
        """

        logits = self._get_independent_bernoulli_logits()  # [batch_shape, lags, num_nodes, num_nodes]
        expanded_logits = logits.expand(*(sample_shape + logits.shape))
        samples = gumbel_softmax_binary(
            logits=expanded_logits, tau=temperature, hard=True
        )  # [sample_shape, batch_shape, lags, num_nodes, num_nodes]

        return samples

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample from the underyling independent Bernoulli distribution.

        Gradients will not flow through this method, use relaxed_sample instead.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (lags, num_nodes, num_nodes)
        """
        logits = self._get_independent_bernoulli_logits()  # [batch_shape, lags, num_nodes, num_nodes]
        samples = self.base_dist(logits).sample(
            sample_shape=sample_shape
        )  # [sample_shape, batch_shape, lags, num_nodes, num_nodes]
        return samples

    def entropy(self) -> torch.Tensor:
        """
        Get the entropy of the independent Bernoulli distribution.

        Returns:
            A tensor of shape batch_shape, with the entropy of the distribution
        """
        logits = self._get_independent_bernoulli_logits()
        # shape of entropy is [batch_shape, lags, num_nodes, num_nodes]
        entropy = F.binary_cross_entropy_with_logits(logits, logits_to_probs(logits, is_binary=True), reduction="none")
        # Entropy should not consider diagonal elements for instantaneous adj matrix
        return torch.sum(entropy, dim=(-3, -2, -1))  # batch_shape

    @property
    def mean(self) -> torch.Tensor:
        """
        Return the mean of the underlying independent Bernoulli distribution.

        This will be a matrix with all entries in the interval [0, 1].

        Returns:
            A tensor of shape batch_shape + (lags, num_nodes, num_nodes)
        """
        logits = self._get_independent_bernoulli_logits()
        # shape of mean is [batch_shape, lags, num_nodes, num_nodes]
        return self.base_dist(logits).mean

    @property
    def mode(self) -> torch.Tensor:
        """
        Return the mode of the underlying independent Bernoulli distribution.

        This will be an adjacency matrix.

        Returns:
            A tensor of shape batch_shape + (lags, num_nodes, num_nodes)
        """
        logits = self._get_independent_bernoulli_logits()
        # bernoulli mode can be nan for very small logits, favour sparseness and set to 0
        # shape of mode is [batch_shape, lags, num_nodes, num_nodes]
        mode = torch.nan_to_num(self.base_dist(logits).mode, nan=0.0)

        return mode

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of each tensor from the sample space

        Args:
            value: a binary matrix of shape value_shape + batch_shape + (lags, n, n)
        Returns:
            A tensor of shape value_shape + batch_shape, with the log probabilities of each tensor in the batch.
        """

        logits = self._get_independent_bernoulli_logits()
        full_log_prob = self.base_dist(logits).log_prob(value)  # value_shape + batch_shape

        return full_log_prob


class RhinoLaggedAdjacencyDistributionModule(DistributionModule[RhinoLaggedAdjacencyDistribution]):
    """Represents an `RhinoLaggedAdjacencyDistributionModule` distribution with learnable parameters."""

    def __init__(self, num_nodes: int, lags: int) -> None:
        """
        Args:
            num_nodes: The number of nodes in the graph. It must be positive.
            lags: The number of lags in the temporal adjacency matrix. It must be positive.
        """
        super().__init__()
        self.logits_edge = nn.Parameter(torch.zeros(lags, num_nodes, num_nodes), requires_grad=True)
        self.lags = lags

    def forward(self) -> RhinoLaggedAdjacencyDistribution:
        return RhinoLaggedAdjacencyDistribution(logits_edge=self.logits_edge, lags=self.lags)


class TemporalAdjacencyDistributionModule(DistributionModule[TemporalAdjacencyDistribution]):
    """Represents an `TemporalAdjacencyDistributionModule` distribution with learnable parameters."""

    def __init__(
        self,
        inst_dist_module: DistributionModule[AdjacencyDistribution],
        lagged_dist_module: Optional[DistributionModule[LaggedAdjacencyDistribution]],
    ) -> None:
        """
        Args:
            inst_dist_module: The distribution module of the instantaneous adjacency matrix.
            lagged_dist_module: The distribution module of the lagged adjacency matrix. If None, we assume only
                instantaneous adjacency exists.
        """
        super().__init__()
        self.inst_dist_module = inst_dist_module
        self.lagged_dist_module = lagged_dist_module

    def forward(self) -> TemporalAdjacencyDistribution:
        return TemporalAdjacencyDistribution(
            instantaneous_distribution=self.inst_dist_module(),
            lagged_distribution=self.lagged_dist_module() if self.lagged_dist_module is not None else None,
        )
