from typing import Optional

import torch
import torch.distributions as td
import torch.nn.functional as F

from causica.distributions.adjacency.adjacency_distributions import AdjacencyDistribution
from causica.triangular_transformations import fill_triangular, num_lower_tri_elements_to_n, unfill_triangular


class ThreeWayAdjacencyDistribution(AdjacencyDistribution):
    """
    For each pair of nodes xᵢ and xⱼ where i < j, sample a three way categorical Cᵢⱼ.
        Cᵢⱼ = 0, represents the edge xᵢ -> xⱼ,
        Cᵢⱼ = 1, represents the edge xᵢ <- xⱼ,
        Cᵢⱼ = 2, there is no edge between these nodes.

    We store the logits of C as a tensor of shape (n(n-1)/2, 3), so we have a three way
    categorical distribution for each edge in the strictly lower triangle.
    """

    arg_constraints = {"logits": td.constraints.real}

    def __init__(self, logits: torch.Tensor, validate_args: Optional[bool] = None):
        """
        Args:
            logits: An array of size (..., n(n-1)/2, 3), representing logit of i->j, j->i, no edge respectively.
            validate_args: Whether to validate the arguments. Passed to the superclass
        """
        num_nodes = num_lower_tri_elements_to_n(logits.shape[-2])

        if validate_args:
            assert len(logits.shape) >= 2, "Logits must be at least dimensional"
            assert logits.shape[-2:] == ((num_nodes * (num_nodes - 1)) // 2, 3), "Invalid logits shape"

        self.logits = logits

        super().__init__(num_nodes, validate_args=validate_args)

    @staticmethod
    def base_dist(logits: torch.Tensor) -> td.Distribution:
        """The underlying distribution is a vector of one hot categorical variables."""
        return td.Independent(td.OneHotCategorical(logits=logits, validate_args=False), 1)

    def relaxed_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0) -> torch.Tensor:
        """Use the Gumbel Softmax trick to relax the distribution."""
        shape = sample_shape + self.logits.shape
        samples = F.gumbel_softmax(
            self.logits.expand(*shape), tau=temperature, hard=True, dim=-1
        )  # (..., n(n-1)/2, 3) binary
        return _triangular_vec_to_matrix(samples)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample a binary adjacency matrix from the underlying distribution.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        return _triangular_vec_to_matrix(self.base_dist(self.logits).sample(sample_shape))

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
        Return the mean of the underlying independent Bernoulli distribution.

        This will be a matrix with all entries in the interval [0, 1].

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        return _triangular_vec_to_matrix(self.base_dist(self.logits).mean)

    @property
    def mode(self) -> torch.Tensor:
        """
        Return the mode of the underlying distribution.

        This will be an adjacency matrix.

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        # bernoulli mode can be nan for very small logits, favour sparseness and set to 0
        return _triangular_vec_to_matrix(torch.nan_to_num(self.base_dist(self.logits).mode, 0.0))

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of each tensor from the sample space

        Args:
            value: a binary matrix of shape value_shape + batch_shape + (n, n)
        Returns:
            A tensor of shape value_shape + batch_shape, with the log probabilities of each tensor in the batch.
        """
        low_tri = unfill_triangular(value, upper=False)
        upp_tri = unfill_triangular(value, upper=True)
        missing = (1 - low_tri) * (1 - upp_tri)
        return self.base_dist(self.logits).log_prob(torch.stack([low_tri, upp_tri, missing], dim=-1))


def _triangular_vec_to_matrix(vec: torch.Tensor) -> torch.Tensor:
    """
    Args:
        vec: A tensor of shape (..., n(n-1)/2, 3)
    Returns:
        A matrix of shape (..., n, n), where the strictly lower triangle is filled from vec[..., 0],
        the strictly upper triangle is filled from vec[..., 1], and zeros on the main diagonal
    """
    return fill_triangular(vec[..., 0], upper=False) + fill_triangular(vec[..., 1], upper=True)
