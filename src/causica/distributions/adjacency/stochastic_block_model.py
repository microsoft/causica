import igraph as ig
import numpy as np
import torch

from causica.distributions.adjacency.adjacency_distributions import AdjacencyDistribution


class StochasticBlockModelDAGDistribution(AdjacencyDistribution):
    """
    An adjacency distribution for sampling Directed Acyclic Graphs using SBM.

    Re-implementation in pytorch from AVICI:

    https://github.com/larslorch/avici


    """

    def __init__(
        self,
        num_nodes: int,
        edges_per_node: list[int],
        num_blocks: list[int],
        damping: list[float],
        validate_args: bool | None = None,
    ):
        """
        Args:
            num_nodes: the number of nodes in the DAGs to be sampled
            edges_per_node: List of the number of edges per node to sample.
            num_blocks: List of the number of blocks in the model
            damp: List of the damp factor for inter block edges. damp = 1.0 is equivalent to erdos renyi
            validate_args: Optional arguments from AdjacencyDistribution
        """
        assert num_nodes > 0, "Number of nodes in the graph must be greater than 0"
        super().__init__(num_nodes=num_nodes, validate_args=validate_args)
        self.edges_per_node = edges_per_node
        self.num_blocks = num_blocks
        self.damping = damping

    def _one_sample(self):
        """
        Sample a binary adjacency matrix from the underlying distribution.

        Returns:
            A tensor of shape (num_nodes, num_nodes)
        """
        get_edges_per_node = np.random.choice(self.edges_per_node)
        get_num_block = np.random.choice(self.num_blocks)
        get_damping = np.random.choice(self.damping)

        # sample blocks
        splits = np.sort(np.random.choice(self.num_nodes, size=get_num_block - 1, replace=False))
        blocks = np.split(np.random.permutation(self.num_nodes), splits)
        block_sizes = np.array([b.shape[0] for b in blocks])

        # select p s.t. we get requested edges_per_var in expectation
        block_edges_sampled = (np.outer(block_sizes, block_sizes) - np.diag(block_sizes)) / 2
        relative_block_probs = np.eye(get_num_block) + get_damping * (1 - np.eye(get_num_block))
        n_edges = get_edges_per_node * self.num_nodes
        p = min(0.99, n_edges / np.sum(block_edges_sampled * relative_block_probs))

        # sample graph
        mat_intra = np.random.binomial(n=1, p=p, size=(self.num_nodes, self.num_nodes)).astype(int)  # bernoulli
        mat_inter = np.random.binomial(n=1, p=get_damping * p, size=(self.num_nodes, self.num_nodes)).astype(
            int
        )  # bernoulli

        mat = np.zeros((self.num_nodes, self.num_nodes))
        for i, bi in enumerate(blocks):
            for j, bj in enumerate(blocks):
                mat[np.ix_(bi, bj)] = (mat_intra if i == j else mat_inter)[np.ix_(bi, bj)]

        # make directed
        dag = np.triu(mat, k=1)

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
