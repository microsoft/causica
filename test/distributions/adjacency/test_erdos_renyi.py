import networkx as nx
import torch

from causica.distributions import ErdosRenyiDAGDistribution


def test_samples_dags():
    """Test that all samples are DAGs"""
    p = torch.tensor([[0.7, 0.4]])
    n = 5
    sample_shape = torch.Size([3, 4])
    dist = ErdosRenyiDAGDistribution(num_nodes=n, probs=p)
    samples = dist.sample(sample_shape).numpy()
    assert samples.shape == torch.Size(sample_shape + p.shape + (n, n))
    flat_samples = samples.reshape((-1, n, n))
    for dag in flat_samples:
        assert nx.is_directed_acyclic_graph(nx.from_numpy_array(dag, create_using=nx.DiGraph))


def test_mode():
    """Test the mode either returns a lower triangle or a matrix of zeros."""
    p = torch.tensor(0.6)
    mode = ErdosRenyiDAGDistribution(num_nodes=5, probs=p).mode
    torch.testing.assert_close(mode, torch.tril(torch.ones_like(mode), diagonal=-1))

    p = torch.tensor(0.2)
    mode = ErdosRenyiDAGDistribution(num_nodes=5, probs=p).mode
    torch.testing.assert_close(mode, torch.zeros_like(mode))


def test_extreme_sample():
    """Test that extreme probabilities give rise to expected graphs"""
    n = 6
    p = torch.tensor(1.0)
    sample = ErdosRenyiDAGDistribution(num_nodes=n, probs=p).sample()
    torch.testing.assert_close(sample.sum(dim=(-2, -1)).item(), n * (n - 1) / 2.0)

    p = torch.tensor(0.0)
    sample = ErdosRenyiDAGDistribution(num_nodes=n, probs=p).sample()
    torch.testing.assert_close(sample, torch.zeros_like(sample))


def test_num_deges():
    num_edges = 16
    samples = ErdosRenyiDAGDistribution(num_nodes=8, num_edges=torch.tensor(num_edges)).sample(torch.Size([100]))

    assert samples.shape == torch.Size([100, 8, 8])
    torch.testing.assert_close(
        samples.sum(dim=(-2, -1)).mean(), torch.tensor(num_edges, dtype=torch.float32), atol=2.0, rtol=0.1
    )

    samples = ErdosRenyiDAGDistribution(num_nodes=2, num_edges=torch.tensor(2)).sample(torch.Size([100]))

    assert samples.shape == torch.Size([100, 2, 2])
    torch.testing.assert_close(samples.sum(dim=(-2, -1)).mean(), torch.tensor(1, dtype=torch.float32))
