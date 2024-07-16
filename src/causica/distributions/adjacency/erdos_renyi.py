import math

import numpy as np
import torch
import torch.distributions as td

from causica.distributions.adjacency.adjacency_distributions import AdjacencyDistribution
from causica.triangular_transformations import fill_triangular


class ErdosRenyiDAGDistribution(AdjacencyDistribution):
    """
    An adjacency distribution for sampling Directed Acyclic Graphs using Erdos Renyi.
    """

    support = td.constraints.independent(td.constraints.boolean, 1)
    arg_constraints = {"probs": td.constraints.unit_interval}

    def __init__(
        self,
        num_nodes: int,
        probs: torch.Tensor | None = None,
        num_edges: torch.Tensor | None = None,
        validate_args: bool | None = None,
    ):
        """
        Args:
            num_nodes: the number of nodes in the DAGs to be sampled
            probs: A tensor of the probability that an edge exists between 2 nodes of shape batch_shape. If None, calculate p from num_edges
            num_edges: The number of edges to sample. If None, sample from a binomial distribution with p=probs
            validate_args: Arguments from AdjacencyDistribution
        """
        assert num_nodes > 0, "Number of nodes in the graph must be greater than 0"

        if num_edges is not None:
            if probs is not None:
                raise ValueError("Only one of probs or num_edges must be provided")
            # Calculate the probability of an edge existing based on the number of edges and the number of possible edges
            probs = torch.minimum(num_edges / math.comb(num_nodes, 2), torch.ones_like(num_edges))
        elif probs is None:
            raise ValueError("Either probs or num_edges must be provided")

        self.probs = probs
        super().__init__(num_nodes=num_nodes, validate_args=validate_args)

        self.num_low_tri = num_nodes * (num_nodes - 1) // 2
        expanded_probs = self.probs[..., None].expand(
            *self.probs.shape + (self.num_low_tri,)
        )  # shape batch_shape + (num_low_tri)
        self.bern_dist = td.Independent(td.Bernoulli(probs=expanded_probs), reinterpreted_batch_ndims=1)
        self.np_rng = np.random.default_rng()

    def relaxed_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample a binary adjacency matrix from the underlying distribution.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        # generate the lower triangle shape [..., n, n]
        low_tri = fill_triangular(self.bern_dist.sample(sample_shape=sample_shape))

        # generate a random permutation matrix
        aranges = np.tile(np.arange(self.num_nodes), sample_shape + self.probs.shape + (1,))  # shape [..., n]
        np_perms = torch.tensor(self.np_rng.permuted(aranges, axis=-1))  # a batch of rearranged [0, 1, 2... n]
        # one hot the last dimension to create a tensor of shape [..., n, n]
        perms = torch.nn.functional.one_hot(np_perms, num_classes=self.num_nodes).to(
            dtype=low_tri.dtype, non_blocking=True
        )

        return torch.einsum("...ij,...jk,...lk->...il", perms, low_tri, perms)

    def entropy(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def mean(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def mode(self) -> torch.Tensor:
        """
        Return the mode of the underlying distribution.

        This will be an adjacency matrix.

        For this distribution, there are many modes since each permutation of nodes is equally likely.
        We return the mode corresponding to the "default" ordering.

        There are 2 possibilities:
            p > 0.5: A lower triangular matrix of ones
            p <= 0.5: A matrix of zeros

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        return fill_triangular(torch.nan_to_num(self.bern_dist.mode, nan=0.0))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
