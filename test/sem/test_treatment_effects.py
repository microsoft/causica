import math

import pytest
import torch
from tensordict import TensorDict

from causica.sem.structural_equation_model import ate, counterfactual, ite

from . import create_lingauss_sem


@pytest.fixture(name="two_variable_dict")
def fixture_two_variable_dict():
    return {"x1": torch.Size([1]), "x2": torch.Size([2])}


@pytest.fixture(name="three_variable_dict")
def fixture_three_variable_dict():
    return {"x1": torch.Size([2]), "x2": torch.Size([2]), "x3": torch.Size([1])}


@pytest.mark.parametrize("graph", [torch.tensor([[0, 0], [1, 0.0]]), torch.tensor([[0, 1], [0, 0.0]])])
def test_ate_ite_cf_two_node(graph, two_variable_dict):
    coef_matrix = torch.rand((3, 3))
    sem = create_lingauss_sem(two_variable_dict, coef_matrix, graph, log_scale=math.log(1e-8))
    intervention_values_a = TensorDict({"x2": torch.tensor([1.42, 0.42])}, batch_size=tuple())
    intervention_values_b = TensorDict({"x2": torch.tensor([0.42, 1.42])}, batch_size=tuple())
    average_treatment_effect = ate(sem, intervention_values_a, intervention_values_b)
    factual_data = sem.sample(torch.Size([100]))
    if graph[0, 1] > 0.0:
        expected_treatment_effect = torch.zeros_like(average_treatment_effect["x1"])
        expected_mean_a = factual_data["x1"]
    else:
        expected_mean_a = torch.einsum("i,ij->j", intervention_values_a["x2"], coef_matrix[1:, :1])
        expected_mean_b = torch.einsum("i,ij->j", intervention_values_b["x2"], coef_matrix[1:, :1])
        expected_treatment_effect = expected_mean_a - expected_mean_b
    assert torch.allclose(average_treatment_effect["x1"], expected_treatment_effect, rtol=1e-4)

    individual_treatment_effect = ite(sem, factual_data, intervention_values_a, intervention_values_b)
    cf_effect = counterfactual(sem, factual_data, intervention_values_a)
    assert torch.allclose(individual_treatment_effect["x1"], expected_treatment_effect)
    assert torch.allclose(cf_effect["x1"], expected_mean_a)


def test_ate_ite_cf_three_node(three_variable_dict):
    """x1->x2->x3"""
    coef_matrix = torch.rand((5, 5))
    graph = torch.zeros(3, 3)
    graph[0, 1] = graph[1, 2] = 1
    sem = create_lingauss_sem(three_variable_dict, coef_matrix, graph, log_scale=math.log(1e-8))
    intervention_values_a = TensorDict({"x2": torch.tensor([1.42, 0.42])}, batch_size=tuple())
    intervention_values_b = TensorDict({"x2": torch.tensor([0.42, 1.42])}, batch_size=tuple())
    average_treatment_effect = ate(sem, intervention_values_a, intervention_values_b)
    assert torch.allclose(average_treatment_effect["x1"], torch.zeros_like(average_treatment_effect["x1"]))
    expected_mean_a = torch.einsum("i,ij->j", intervention_values_a["x2"], coef_matrix[2:4, 4:])
    expected_mean_b = torch.einsum("i,ij->j", intervention_values_b["x2"], coef_matrix[2:4, 4:])
    expected_treatment_effect = expected_mean_a - expected_mean_b
    assert torch.allclose(average_treatment_effect["x3"], expected_treatment_effect)

    factual_data = sem.sample(torch.Size([100]))
    individual_treatment_effect = ite(sem, factual_data, intervention_values_a, intervention_values_b)
    cf_effect = counterfactual(sem, factual_data, intervention_values_a)
    assert torch.allclose(individual_treatment_effect["x1"], torch.zeros_like(individual_treatment_effect["x1"]))
    assert torch.allclose(individual_treatment_effect["x3"], expected_treatment_effect)
    assert torch.allclose(cf_effect["x1"], factual_data["x1"])
    assert torch.allclose(cf_effect["x3"], expected_mean_a)
