import math

import pytest
import torch

from causica.functional_relationships import (
    DECIEmbedFunctionalRelationships,
    LinearFunctionalRelationships,
    RFFFunctionalRelationships,
)


@pytest.fixture(name="two_variable_dict")
def fixture_two_variable_dict():
    return {"x1": torch.Size([1]), "x2": torch.Size([2])}


@pytest.fixture(name="two_variable_sample")
def fixture_two_variable_sample():
    return {"x1": torch.randn((3, 1)), "x2": torch.randn((3, 2))}


@pytest.fixture(name="two_variable_graph")
def fixture_two_variable_graph():
    return torch.Tensor([[0.0, 1.0], [0.0, 0.0]])


@pytest.fixture(name="two_variable_graphs")
def fixture_two_variable_graphs():
    return torch.Tensor([[[0.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]])


def test_func_rel_init(two_variable_dict):
    func_rel = DECIEmbedFunctionalRelationships(two_variable_dict, 32, 32, 2, 2)

    assert func_rel.tensor_to_td.output_shape == 3


def test_func_rel_forward(two_variable_dict, two_variable_graph, two_variable_sample):
    func_rel = DECIEmbedFunctionalRelationships(two_variable_dict, 32, 32, 2, 2)
    func_rel.nn.w = torch.nn.Parameter(torch.ones_like(func_rel.nn.w), requires_grad=False)
    prediction = func_rel(two_variable_sample, two_variable_graph)
    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2)
    assert torch.allclose(prediction["x1"], prediction["x1"][0, 0])
    assert not torch.allclose(prediction["x2"][..., 0], prediction["x2"][0, 0])
    assert not torch.allclose(prediction["x2"][..., 1], prediction["x2"][0, 1])


def test_func_rel_forward_multigraph(two_variable_dict, two_variable_graphs, two_variable_sample):
    func_rel = DECIEmbedFunctionalRelationships(two_variable_dict, 32, 32, 2, 2)
    func_rel.nn.w = torch.nn.Parameter(torch.ones_like(func_rel.nn.w), requires_grad=False)
    prediction = func_rel(two_variable_sample, two_variable_graphs)
    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 2, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2, 2)
    # x1 and x2 are initial nodes for graphs 0 and 1 respectively, so should just get bias terms
    assert torch.allclose(prediction["x1"][:, 0, :], prediction["x1"][0, 0, :])
    assert torch.allclose(prediction["x2"][:, 1, :], prediction["x2"][0, 1, :])

    prediction2 = func_rel({"x1": torch.randn((3, 1)), "x2": torch.randn((3, 2))}, two_variable_graphs)
    assert torch.allclose(prediction2["x1"][:, 0, :], prediction["x1"][:, 0, :])
    assert torch.allclose(prediction2["x2"][:, 1, :], prediction["x2"][:, 1, :])

    # for the other graphs they shouldn't be equal
    assert not torch.allclose(prediction2["x1"][:, 1, :], prediction["x1"][:, 1, :])
    assert not torch.allclose(prediction2["x2"][:, 0, :], prediction["x2"][:, 0, :])


def test_linear_forward(two_variable_dict, two_variable_graph, two_variable_sample):
    coef_matrix = torch.rand((3, 3))
    func_rel = LinearFunctionalRelationships(two_variable_dict, coef_matrix)
    prediction = func_rel(two_variable_sample, two_variable_graph)
    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2)
    assert torch.all(prediction["x1"] == 0.0)
    true_x2_prediction = torch.matmul(two_variable_sample["x1"].unsqueeze(-2), coef_matrix[:1, 1:]).squeeze(-2)
    assert torch.all(prediction["x2"] == true_x2_prediction)


def test_linear_forward_multigraph(two_variable_dict, two_variable_graphs, two_variable_sample):
    coef_matrix = torch.rand((3, 3))
    func_rel = LinearFunctionalRelationships(two_variable_dict, coef_matrix)
    prediction = func_rel(two_variable_sample, two_variable_graphs)

    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 2, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2, 2)

    # x1 and x2 are initial nodes for graphs 0 and 1 respectively
    assert torch.all(prediction["x1"][:, 0, :] == 0.0)
    assert torch.all(prediction["x2"][:, 1, :] == 0.0)

    # x1 and x2 are linear transformations of the other nodes for graphs 1 and 0 respectively
    true_x1_prediction = torch.matmul(two_variable_sample["x2"], coef_matrix[1:, :1])
    assert torch.allclose(prediction["x1"][:, 1, :], true_x1_prediction)
    true_x2_prediction = torch.matmul(two_variable_sample["x1"], coef_matrix[:1, 1:])
    assert torch.allclose(prediction["x2"][:, 0, :], true_x2_prediction)


def test_non_linear_forward(two_variable_dict, two_variable_graph, two_variable_sample):
    random_features = torch.rand((5, 3))
    coeff_alpha = torch.rand((5,))
    func_rel = RFFFunctionalRelationships(two_variable_dict, random_features, coeff_alpha)
    prediction = func_rel(two_variable_sample, two_variable_graph)
    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2)
    assert torch.all(prediction["x1"] == 0.0)

    true_inner_prods = two_variable_sample["x1"] * random_features[:, 0]
    transformed_inner_prods = torch.sin(true_inner_prods) * coeff_alpha
    true_x2_prediction = math.sqrt(2 / 5) * torch.sum(transformed_inner_prods, dim=-1)
    true_x2_prediction = true_x2_prediction.unsqueeze(-1).repeat(1, 2)
    assert torch.allclose(prediction["x2"], true_x2_prediction)


def test_non_linear_forward_multigraph(two_variable_dict, two_variable_graphs, two_variable_sample):
    random_features = torch.rand((5, 3))
    coeff_alpha = torch.rand((5,))
    func_rel = RFFFunctionalRelationships(two_variable_dict, random_features, coeff_alpha)
    prediction = func_rel(two_variable_sample, two_variable_graphs)

    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 2, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2, 2)

    # x1 and x2 are initial nodes for graphs 0 and 1 respectively
    assert torch.all(prediction["x1"][:, 0, :] == 0.0)
    assert torch.all(prediction["x2"][:, 1, :] == 0.0)

    true_inner_prods = two_variable_sample["x1"] * random_features[:, 0]
    transformed_inner_prods = torch.sin(true_inner_prods) * coeff_alpha
    true_x2_prediction = math.sqrt(2 / 5) * torch.sum(transformed_inner_prods, dim=-1)
    true_x2_prediction = true_x2_prediction.unsqueeze(-1).repeat(1, 2)
    assert torch.allclose(prediction["x2"][:, 0, :], true_x2_prediction)

    true_inner_prods = torch.matmul(two_variable_sample["x2"], random_features[:, 1:].transpose(-2, -1))
    transformed_inner_prods = torch.sin(true_inner_prods) * coeff_alpha
    true_x1_prediction = math.sqrt(2 / 5) * torch.sum(transformed_inner_prods, dim=-1)
    true_x1_prediction = true_x1_prediction.unsqueeze(-1)
    assert torch.allclose(prediction["x1"][:, 1, :], true_x1_prediction)
