import torch

from causica.distributions.adjacency.adjacency_distributions import AdjacencyDistribution


class FixedAdjacencyDistribution(AdjacencyDistribution):
    """Delta distribution over a fixed adjacency matrix."""

    def __init__(self, adjacency: torch.Tensor, validate_args: bool | None = None):
        if adjacency.shape[-1] != adjacency.shape[-2]:
            raise ValueError("Adjacency matrix must be square")
        if adjacency.ndim != 2:
            raise ValueError("Adjacency matrix must be 2 dimensional")

        self.adjacency = adjacency
        self.num_nodes = adjacency.shape[-1]
        super().__init__(num_nodes=self.num_nodes, validate_args=validate_args)

    def relaxed_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0) -> torch.Tensor:
        """
        Return the underlying adjacency matrix.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        _ = temperature
        return self.sample(sample_shape=sample_shape)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Return the underlying adjacency matrix.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        return self.adjacency.expand(sample_shape + self.adjacency.shape)

    def entropy(self) -> torch.Tensor:

        return torch.zeros([])

    @property
    def mean(self) -> torch.Tensor:
        return self.adjacency

    @property
    def mode(self) -> torch.Tensor:
        return self.adjacency

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
