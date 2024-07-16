import igraph as ig
import numpy as np
import torch

from causica.distributions.adjacency.adjacency_distributions import AdjacencyDistribution


class ScaleFreeDAGDistribution(AdjacencyDistribution):
    """
    An adjacency distribution for sampling Directed Acyclic Graphs using Barabasi Albert.

    Re-implementation in pytorch from AVICI:

    https://github.com/larslorch/avici

    """

    def __init__(
        self,
        num_nodes: int,
        edges_per_node: list[int],
        power: list[float],
        in_degree: bool = True,
        validate_args: bool | None = None,
    ):
        """
        Args:
            num_nodes: the number of nodes in the DAGs to be sampled
            edges_per_node: List of the number of edges per node to sample.
            power: List of the power of the scale free distribution.
            in_degree: If True, the in-degree of each node is sampled from the power law distribution.
                Otherwise, the out-degree is sampled.
            validate_args: Optional arguments from AdjacencyDistribution
        """
        assert num_nodes > 0, "Number of nodes in the graph must be greater than 0"
        super().__init__(num_nodes=num_nodes, validate_args=validate_args)
        self.edges_per_node = edges_per_node
        self.power = power
        self.in_degree = in_degree

    def _one_sample(self):
        """
        Sample a binary adjacency matrix from the underlying distribution.

        Returns:
            A tensor of shape (num_nodes, num_nodes)
        """

        get_edges_per_node = np.random.choice(self.edges_per_node).item()
        get_power = np.random.choice(self.power).item()
        perm = np.random.permutation(self.num_nodes).tolist()
        graph = ig.Graph.Barabasi(
            n=self.num_nodes, m=get_edges_per_node, directed=True, power=get_power
        ).permute_vertices(perm)
        dag = np.array(graph.get_adjacency().data).astype(int)
        if not self.in_degree:
            dag = dag.T

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
