import networkx as nx
import torch

from causica.distributions import ScaleFreeDAGDistribution


def test_samples_dags():
    """Test that all samples are DAGs"""
    edges_per_node = [1, 2]
    power = [1.0, 3.0]
    n = 5
    sample_shape = torch.Size([3, 4])
    dist = ScaleFreeDAGDistribution(num_nodes=n, edges_per_node=edges_per_node, power=power)
    samples = dist.sample(sample_shape).numpy()
    assert samples.shape == torch.Size(sample_shape + (n, n))
    flat_samples = samples.reshape((-1, n, n))
    for dag in flat_samples:
        assert nx.is_directed_acyclic_graph(nx.from_numpy_array(dag, create_using=nx.DiGraph))


def test_num_edges():
    n = 8
    edges_per_node = [2]
    power = [1.0]
    num_edges_expected = edges_per_node[0] * n
    samples = ScaleFreeDAGDistribution(num_nodes=n, edges_per_node=edges_per_node, power=power).sample(
        torch.Size([100])
    )

    assert samples.shape == torch.Size([100, 8, 8])
    torch.testing.assert_close(
        samples.sum(dim=(-2, -1)).mean(), torch.tensor(num_edges_expected, dtype=torch.float32), atol=2.0, rtol=0.1
    )


def test_out_degree():
    n = 8
    edges_per_node = [2]
    power = [1.0]
    in_degree = False
    dag = ScaleFreeDAGDistribution(
        num_nodes=n, edges_per_node=edges_per_node, power=power, in_degree=in_degree
    ).sample()
    assert dag.shape == torch.Size([8, 8])
    dag = dag.numpy()
    assert nx.is_directed_acyclic_graph(nx.from_numpy_array(dag, create_using=nx.DiGraph))
