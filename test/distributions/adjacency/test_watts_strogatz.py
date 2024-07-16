import networkx as nx
import torch

from causica.distributions import WattsStrogatzDAGDistribution


def test_samples_dags():
    """Test that all samples are DAGs"""
    n = 5
    lattice_dim = [2, 3]
    rewire_prob = [0.1, 0.3]
    neighbors = [1, 3]
    sample_shape = torch.Size([3, 4])
    dist = WattsStrogatzDAGDistribution(
        num_nodes=n, lattice_dim=lattice_dim, rewire_prob=rewire_prob, neighbors=neighbors
    )
    samples = dist.sample(sample_shape).numpy()
    assert samples.shape == torch.Size(sample_shape + (n, n))
    flat_samples = samples.reshape((-1, n, n))
    for dag in flat_samples:
        assert nx.is_directed_acyclic_graph(nx.from_numpy_array(dag, create_using=nx.DiGraph))
