import pytest
import torch
import torch.distributions as td
from tensordict import TensorDict

from causica.distributions import NoiseAccessibleBernoulli, NoiseAccessibleIndependent
from causica.functional_relationships import LinearFunctionalRelationships
from causica.sem.distribution_parameters_sem import DistributionParametersSEM

from . import create_lingauss_sem


@pytest.fixture(name="two_variable_dict")
def fixture_two_variable_dict():
    return {"x1": torch.Size([1]), "x2": torch.Size([2])}


@pytest.fixture(name="three_variable_dict")
def fixture_three_variable_dict():
    return {"x1": torch.Size([2]), "x2": torch.Size([2]), "x3": torch.Size([1])}


@pytest.mark.parametrize("graph", [torch.tensor([[0, 0], [1, 0.0]]), torch.tensor([[0, 1], [0, 0.0]])])
def test_do_linear_sem(graph, two_variable_dict):
    coef_matrix = torch.rand((3, 3))
    sem = create_lingauss_sem(two_variable_dict, coef_matrix, graph)
    intervention_value = torch.tensor([1.42, 0.42])
    do_sem = sem.do(TensorDict({"x2": intervention_value}, batch_size=tuple()))
    array = torch.linspace(-2, 2, 100).unsqueeze(-1)
    log_probs = do_sem.log_prob(TensorDict({"x1": array}, batch_size=[100]))
    if graph[1, 0] == 1.0:
        expected_mean = torch.einsum("i,ij->j", intervention_value, coef_matrix[1:, :1])
    else:
        expected_mean = torch.tensor([0.0])
    expected_log_probs = td.Independent(td.Normal(expected_mean, 1.0), 1).log_prob(array)
    assert torch.allclose(log_probs, expected_log_probs)
    noise = sem.sample_noise((10,))
    do_sample = do_sem.noise_to_sample(noise)
    sample = sem.noise_to_sample(noise)
    if graph[1, 0] == 1.0:
        assert torch.allclose(do_sample["x1"], expected_mean + noise["x1"])
    else:
        assert torch.allclose(do_sample["x1"], sample["x1"])


def test_linear_sem_3d_graph_do_1_node(three_variable_dict):
    coef_matrix = torch.rand((5, 5))
    graph = torch.zeros(3, 3)
    graph[0, 1] = graph[0, 2] = 1
    sem = create_lingauss_sem(three_variable_dict, coef_matrix, graph)
    intervention_value = torch.tensor([1.42, 0.42])
    do_sem = sem.do(TensorDict({"x1": intervention_value}, batch_size=tuple()))
    test_val = torch.rand(100, 3)
    log_probs = do_sem.log_prob(TensorDict({"x2": test_val[:, 0:2], "x3": test_val[:, 2:]}, batch_size=[100]))
    expected_mean = torch.einsum("i,ij->j", intervention_value, coef_matrix[:2, 2:])
    expected_log_probs = td.Independent(td.Normal(expected_mean, 1.0), 1).log_prob(test_val)
    assert torch.allclose(log_probs, expected_log_probs)


def test_linear_sem_3d_graph_do_2_nodes(three_variable_dict):
    coef_matrix = torch.rand((5, 5))
    graph = torch.triu(torch.ones(3, 3), diagonal=1)
    sem = create_lingauss_sem(three_variable_dict, coef_matrix, graph)
    int_data = TensorDict({"x1": torch.tensor([1.42, 0.42]), "x2": torch.tensor([-0.42, 0.402])}, batch_size=tuple())
    do_sem = sem.do(int_data)
    test_val = torch.linspace(-2, 2, 100).unsqueeze(-1)
    log_probs = do_sem.log_prob(TensorDict({"x3": test_val}, batch_size=[100]))
    expected_mean = torch.einsum("i,ij->j", torch.cat((int_data["x1"], int_data["x2"])), coef_matrix[:4, 4:])
    expected_log_probs = td.Independent(td.Normal(expected_mean, 1.0), 1).log_prob(test_val)
    assert torch.allclose(log_probs, expected_log_probs)


@pytest.mark.parametrize("graph", [torch.tensor([[0, 0], [1, 0.0]]), torch.tensor([[0, 1], [0, 0.0]])])
def test_do_linear_sem_bernoulli(graph, two_variable_dict):
    coef_matrix = torch.rand((3, 3))
    func = LinearFunctionalRelationships(two_variable_dict, initial_linear_coefficient_matrix=coef_matrix)
    noise_dist = {
        key: lambda x, val=val: NoiseAccessibleIndependent(
            NoiseAccessibleBernoulli(delta_logits=x, base_logits=torch.zeros(val)), 1
        )
        for key, val in two_variable_dict.items()
    }
    sem = DistributionParametersSEM(graph=graph, node_names=two_variable_dict.keys(), noise_dist=noise_dist, func=func)
    intervention_value = torch.tensor([1.42, 0.42])
    do_sem = sem.do(TensorDict({"x2": intervention_value}, batch_size=tuple()))
    array = torch.bernoulli(0.5 * torch.ones(100, 1))
    log_probs = do_sem.log_prob(TensorDict({"x1": array}, batch_size=array.shape[:-1]))
    if graph[1, 0] == 1.0:
        expected_logits = torch.einsum("i,ij->j", intervention_value, coef_matrix[1:, :1])
    else:
        expected_logits = torch.tensor([0.0])
    expected_log_probs = td.Independent(td.Bernoulli(logits=expected_logits), 1).log_prob(array)
    torch.testing.assert_close(log_probs, expected_log_probs)
    noise = sem.sample_noise((10,))
    do_sample = do_sem.noise_to_sample(noise)
    sample = sem.noise_to_sample(noise)
    if graph[1, 0] == 1.0:
        torch.testing.assert_close(do_sample["x1"], ((expected_logits + noise["x1"]) > 0).float())
    else:
        torch.testing.assert_close(do_sample["x1"], sample["x1"])


def test_linear_sem_3d_graph_do_1_node_bernoulli(three_variable_dict):
    coef_matrix = torch.rand((5, 5))
    graph = torch.zeros(3, 3)
    graph[0, 1] = graph[0, 2] = 1
    func = LinearFunctionalRelationships(three_variable_dict, coef_matrix)
    noise_dist = {
        key: lambda x, val=val: NoiseAccessibleIndependent(
            NoiseAccessibleBernoulli(delta_logits=x, base_logits=torch.zeros(val)), 1
        )
        for key, val in three_variable_dict.items()
    }
    sem = DistributionParametersSEM(
        graph=graph, node_names=three_variable_dict.keys(), noise_dist=noise_dist, func=func
    )
    intervention_value = torch.tensor([1.42, 0.42])
    do_sem = sem.do(TensorDict({"x1": intervention_value}, batch_size=tuple()))
    array = torch.bernoulli(0.5 * torch.ones(100, 3))
    log_probs = do_sem.log_prob(TensorDict({"x2": array[:, :2], "x3": array[:, 2:]}, batch_size=array.shape[:-1]))
    expected_logits = torch.einsum("i,ij->j", intervention_value, coef_matrix[:2, 2:])
    expected_log_probs = td.Independent(td.Bernoulli(logits=expected_logits), 1).log_prob(array)
    torch.testing.assert_close(log_probs, expected_log_probs)


def test_linear_sem_3d_graph_do_2_nodes_bernoulli(three_variable_dict):
    coef_matrix = torch.rand((5, 5))
    graph = torch.triu(torch.ones(3, 3), diagonal=1)
    func = LinearFunctionalRelationships(three_variable_dict, coef_matrix)
    noise_dist = {
        key: lambda x, val=val: NoiseAccessibleIndependent(
            NoiseAccessibleBernoulli(delta_logits=x, base_logits=torch.zeros(val)), 1
        )
        for key, val in three_variable_dict.items()
    }
    sem = DistributionParametersSEM(
        graph=graph, node_names=three_variable_dict.keys(), noise_dist=noise_dist, func=func
    )
    int_data = TensorDict({"x1": torch.tensor([1.42, 0.42]), "x2": torch.tensor([-0.42, 0.402])}, batch_size=tuple())
    do_sem = sem.do(int_data)
    array = torch.bernoulli(0.5 * torch.ones(100, 1))
    log_probs = do_sem.log_prob(TensorDict({"x3": array}, batch_size=array.shape[:-1]))
    expected_logits = torch.einsum("i,ij->j", torch.cat((int_data["x1"], int_data["x2"])), coef_matrix[:4, 4:])
    expected_log_probs = td.Independent(td.Bernoulli(logits=expected_logits), 1).log_prob(array)
    torch.testing.assert_close(log_probs, expected_log_probs)
