import math

import igraph as ig
import numpy as np
import torch

from causica.distributions.adjacency.adjacency_distributions import AdjacencyDistribution


class WattsStrogatzDAGDistribution(AdjacencyDistribution):
    """
    An adjacency distribution for sampling Directed Acyclic Graphs using Watts Strogatz.

    Re-implementation in pytorch from AVICI:

    https://github.com/larslorch/avici

    """

    def __init__(
        self,
        num_nodes: int,
        lattice_dim: list[int],
        rewire_prob: list[float],
        neighbors: list[int],
        validate_args: bool | None = None,
    ):
        """
        Args:
            num_nodes: the number of nodes in the DAGs to be sampled
            lattice_dim: List of the dimension of the lattice
            rewire_prob: List of the rewiring probability
            neighbors: List of the number of neighbors used
            validate_args: Optional arguments from AdjacencyDistribution
        """
        assert num_nodes > 0, "Number of nodes in the graph must be greater than 0"
        super().__init__(num_nodes=num_nodes, validate_args=validate_args)
        self.lattice_dim = lattice_dim
        self.rewire_prob = rewire_prob
        self.neighbors = neighbors

    def _one_sample(self):
        """
        Sample a binary adjacency matrix from the underlying distribution.

        Returns:
            A tensor of shape (num_nodes, num_nodes)
        """
        get_lattice_dim = np.random.choice(self.lattice_dim).item()
        # choose size s.t. we get at smallest possible n_vars greater than requested n_vars given the dimension of lattice
        dim_size = math.ceil(self.num_nodes ** (1.0 / get_lattice_dim))
        get_rewire_prob = np.random.choice(self.rewire_prob).item()
        get_neighbors = np.random.choice(self.neighbors).item()
        graph = ig.Graph.Watts_Strogatz(
            dim=get_lattice_dim, size=dim_size, nei=get_neighbors, p=get_rewire_prob, multiple=False, loops=False
        )

        # drop excessive vertices s.t. we get exactly n_vars
        n_excessive = len(graph.vs) - self.num_nodes
        assert n_excessive >= 0
        if n_excessive:
            graph.delete_vertices(np.random.choice(graph.vs, size=n_excessive, replace=False))
        assert (
            len(graph.vs) == self.num_nodes
        ), f"Didn't get requested graph; g.vs: {len(graph.vs)}, n_vars {self.num_nodes}"

        # make directed
        dag = np.triu(np.array(graph.get_adjacency().data).astype(int), k=1)

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
