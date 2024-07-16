import networkx as nx
import torch

from causica.distributions import GeometricRandomGraphDAGDistribution


def test_samples_dags():
    """Test that all samples are DAGs"""
    radius = [0.1, 0.5]
    n = 5
    sample_shape = torch.Size([3, 4])
    dist = GeometricRandomGraphDAGDistribution(num_nodes=n, radius=radius)
    samples = dist.sample(sample_shape).numpy()
    assert samples.shape == torch.Size(sample_shape + (n, n))
    flat_samples = samples.reshape((-1, n, n))
    for dag in flat_samples:
        assert nx.is_directed_acyclic_graph(nx.from_numpy_array(dag, create_using=nx.DiGraph))
