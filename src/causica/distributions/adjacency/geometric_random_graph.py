import igraph as ig
import numpy as np
import torch

from causica.distributions.adjacency.adjacency_distributions import AdjacencyDistribution


class GeometricRandomGraphDAGDistribution(AdjacencyDistribution):
    """
    An adjacency distribution for sampling Directed Acyclic Graphs using GRG.

    Re-implementation in pytorch from AVICI:

    https://github.com/larslorch/avici

    """

    def __init__(
        self,
        num_nodes: int,
        radius: list[float],
        validate_args: bool | None = None,
    ):
        """
        Args:
            num_nodes: the number of nodes in the DAGs to be sampled
            radius: List of the radius of the geometric random graph
            validate_args: Optional arguments from AdjacencyDistribution
        """
        assert num_nodes > 0, "Number of nodes in the graph must be greater than 0"
        super().__init__(num_nodes=num_nodes, validate_args=validate_args)
        self.radius = radius

    def _one_sample(self):
        """
        Sample a binary adjacency matrix from the underlying distribution.

        Returns:
            A tensor of shape (num_nodes, num_nodes)
        """
        get_radius = np.random.choice(self.radius).item()
        perm = np.random.permutation(self.num_nodes).tolist()
        graph = ig.Graph.GRG(n=self.num_nodes, radius=get_radius)
        graph = np.array(graph.get_adjacency().data).astype(int)

        dag = np.triu(graph, k=1)
        dag = dag[perm, :][:, perm]

        if not ig.Graph.Weighted_Adjacency(dag.tolist()).is_dag():
            raise ValueError("Sampled graph is not a DAG")

        dag = torch.tensor(dag, dtype=torch.float32)
        return dag

    def relaxed_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, sample_shape: torch.Size = torch.Size()):
        """
        Sample binary adjacency matrices from the underlying distribution.

        Args:
            sample_shape: The shape of the samples to be drawn.
        """
        num_samples = int(torch.prod(torch.tensor(sample_shape)).item())
        return torch.stack([self._one_sample() for _ in range(num_samples)]).reshape(
            sample_shape + (self.num_nodes, self.num_nodes)
        )

    def entropy(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def mean(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def mode(self) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
