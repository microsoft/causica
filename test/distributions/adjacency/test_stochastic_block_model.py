import networkx as nx
import torch

from causica.distributions import StochasticBlockModelDAGDistribution


def test_samples_dags():
    """Test that all samples are DAGs"""
    edges_per_node = [1, 2]
    n = 5
    num_blocks = [2, 3]
    damping = [0.1, 0.2]
    sample_shape = torch.Size([3, 4])
    dist = StochasticBlockModelDAGDistribution(
        num_nodes=n, edges_per_node=edges_per_node, num_blocks=num_blocks, damping=damping
    )
    samples = dist.sample(sample_shape).numpy()
    assert samples.shape == torch.Size(sample_shape + (n, n))
    flat_samples = samples.reshape((-1, n, n))
    for dag in flat_samples:
        assert nx.is_directed_acyclic_graph(nx.from_numpy_array(dag, create_using=nx.DiGraph))
