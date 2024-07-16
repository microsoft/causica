import igraph as ig
import numpy as np
import torch

from causica.distributions.adjacency.adjacency_distributions import AdjacencyDistribution


class EdgesPerNodeErdosRenyiDAGDistribution(AdjacencyDistribution):
    """
    An adjacency distribution for sampling Directed Acyclic Graphs using Erdos-Renyi.

    It allows to sample DAGs according to the ER distribution with a fixed number of edges per node.

    Re-implementation in pytorch from AVICI:

    https://github.com/larslorch/avici

    """

    def __init__(
        self,
        num_nodes: int,
        edges_per_node: list[int],
        validate_args: bool | None = None,
    ):
        """
        Args:
            num_nodes: the number of nodes in the DAGs to be sampled
            edges_per_node: List of the number of edges per node to sample.
        """
        assert num_nodes > 0, "Number of nodes in the graph must be greater than 0"
        super().__init__(num_nodes=num_nodes, validate_args=validate_args)
        self.edges_per_node = edges_per_node

    def _one_sample(self):
        """
        Sample a binary adjacency matrix from the underlying distribution.

        Returns:
            A tensor of shape (num_nodes, num_nodes)
        """

        get_edges_per_node = np.random.choice(self.edges_per_node).item()
        n_edges = get_edges_per_node * self.num_nodes
        prob = min(n_edges / ((self.num_nodes * (self.num_nodes - 1)) / 2), 0.99)

        mat = np.random.binomial(n=1, p=prob, size=(self.num_nodes, self.num_nodes)).astype(int)  # bernoulli

        # make DAG by zeroing above diagonal; k=-1 indicates that diagonal is zero too
        dag = np.tril(mat, k=-1)

        # randomly permute
        perm = np.random.permutation(self.num_nodes).tolist()
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

        Returns:
            A tensor of shape (sample_shape, num_nodes, num_nodes)
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
