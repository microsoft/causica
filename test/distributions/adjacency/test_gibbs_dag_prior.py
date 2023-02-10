import pytest
import torch

from causica.distributions.adjacency.gibbs_dag_prior import ExpertGraphContainer, GibbsDAGPrior


def test_expert_graph_dataclass():

    mask = torch.Tensor([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    dag = torch.Tensor([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    confidence = 0.8
    scale = 10

    expert_graph_container = ExpertGraphContainer(dag, mask, confidence, scale)

    assert expert_graph_container.mask.shape[0] == 3
    assert expert_graph_container.mask.shape[1] == 3

    assert expert_graph_container.dag.shape[0] == 3
    assert expert_graph_container.dag.shape[1] == 3

    assert expert_graph_container.confidence == 0.8


def test_get_sparsity_term():
    gibbs_dag_prior = GibbsDAGPrior(num_nodes=2, sparsity_lambda=torch.tensor(1))

    dag = torch.Tensor([[0, 0], [0, 0]])

    assert gibbs_dag_prior.get_sparsity_term(dag) == 0

    dense_dag = torch.Tensor([[1, 1], [1, 1]])

    sparse_dag = torch.Tensor([[0, 1], [0, 1]])

    assert gibbs_dag_prior.get_sparsity_term(dense_dag) > gibbs_dag_prior.get_sparsity_term(sparse_dag)


def test_get_expert_graph_term():

    mask = torch.Tensor([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    dag = torch.Tensor([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    confidence = 0.8
    scale = 10

    expert_graph_container = ExpertGraphContainer(dag, mask, confidence, scale)

    gibbs_dag_prior = GibbsDAGPrior(
        num_nodes=3,
        sparsity_lambda=torch.tensor(1),
        expert_graph_container=expert_graph_container,
    )

    A = torch.Tensor([[0, 0, 1], [0, 1, 1], [1, 0, 1]])

    assert gibbs_dag_prior.get_expert_graph_term(A) == torch.tensor(0)

    mask = torch.Tensor([[0, 0, 1], [0, 0, 0], [0, 0, 0]])

    expert_graph_container = ExpertGraphContainer(dag, mask, confidence, scale)
    gibbs_dag_prior = GibbsDAGPrior(
        num_nodes=3,
        sparsity_lambda=torch.tensor(1),
        expert_graph_container=expert_graph_container,
    )

    torch.testing.assert_close(gibbs_dag_prior.get_expert_graph_term(A), torch.tensor(0.2))


def test_log_prob():
    gibbs_dag_prior = GibbsDAGPrior(num_nodes=123, sparsity_lambda=torch.tensor(1))

    A = torch.Tensor([[0, 0, 1], [0, 0, 1], [0, 0, 0]])

    with pytest.raises(AssertionError):
        gibbs_dag_prior.log_prob(A)

    gibbs_dag_prior = GibbsDAGPrior(num_nodes=2, sparsity_lambda=torch.tensor(1))

    A = torch.Tensor([[1, 1], [0, 1]])

    torch.testing.assert_close(
        gibbs_dag_prior.log_prob(A),
        torch.tensor(-3.0),
    )

    A = torch.Tensor([[0, 1], [0, 0]])

    torch.testing.assert_close(gibbs_dag_prior.log_prob(A), torch.tensor(-1.0))
