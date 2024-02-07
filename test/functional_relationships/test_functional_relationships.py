import math

import pytest
import torch
from tensordict import TensorDict

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
    return TensorDict({"x1": torch.randn((3, 1)), "x2": torch.randn((3, 2))}, batch_size=torch.Size([3]))


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
    prediction = func_rel(two_variable_sample.unsqueeze(1).expand(3, 2), two_variable_graphs)
    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 2, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2, 2)
    # x1 and x2 are initial nodes for graphs 0 and 1 respectively, so should just get bias terms
    assert torch.allclose(prediction["x1"][:, 0, :], prediction["x1"][0, 0, :])
    assert torch.allclose(prediction["x2"][:, 1, :], prediction["x2"][0, 1, :])

    # Pass a new sample through the model and check that the predictions are the same for the initial nodes
    sample2 = TensorDict({"x1": torch.randn((3, 1)), "x2": torch.randn((3, 2))}, batch_size=torch.Size([3]))
    prediction2 = func_rel(sample2.unsqueeze(1).expand(3, 2), two_variable_graphs)
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
    prediction = func_rel(two_variable_sample.unsqueeze(1).expand(3, 2), two_variable_graphs)

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


def test_linear_forward_with_bias(two_variable_dict, two_variable_graph, two_variable_sample):
    coef_matrix = torch.rand((3, 3))
    initial_bias = torch.rand((3,))
    func_rel = LinearFunctionalRelationships(two_variable_dict, coef_matrix, initial_bias=initial_bias)
    prediction = func_rel(two_variable_sample, two_variable_graph)
    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2)
    assert torch.all(prediction["x1"] == initial_bias[0])
    true_x2_prediction = initial_bias[1:] + torch.matmul(
        two_variable_sample["x1"].unsqueeze(-2), coef_matrix[:1, 1:]
    ).squeeze(-2)
    assert torch.all(prediction["x2"] == true_x2_prediction)


def test_linear_forward_multigraph_with_bias(two_variable_dict, two_variable_graphs, two_variable_sample):
    coef_matrix = torch.rand((3, 3))
    initial_bias = torch.rand((3,))
    func_rel = LinearFunctionalRelationships(two_variable_dict, coef_matrix, initial_bias=initial_bias)
    prediction = func_rel(two_variable_sample.unsqueeze(1).expand(3, 2), two_variable_graphs)

    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 2, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2, 2)

    # x1 and x2 are initial nodes for graphs 0 and 1 respectively
    assert torch.allclose(prediction["x1"][:, 0, :], initial_bias[0].repeat(3, 1))
    assert torch.allclose(prediction["x2"][:, 1, :], initial_bias[1:].repeat(3, 1))

    # x1 and x2 are linear transformations of the other nodes for graphs 1 and 0 respectively
    true_x1_prediction = initial_bias[0] + torch.matmul(two_variable_sample["x2"], coef_matrix[1:, :1])
    assert torch.allclose(prediction["x1"][:, 1, :], true_x1_prediction)
    true_x2_prediction = initial_bias[1:].reshape(1, -1) + torch.matmul(two_variable_sample["x1"], coef_matrix[:1, 1:])
    assert torch.allclose(prediction["x2"][:, 0, :], true_x2_prediction)


def test_non_linear_forward(two_variable_dict, two_variable_graph, two_variable_sample):
    random_features = torch.rand((5, 3))
    coeff_alpha = torch.rand((5,))
    func_rel = RFFFunctionalRelationships(two_variable_dict, random_features, coeff_alpha)
    prediction = func_rel(two_variable_sample, two_variable_graph)
    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2)
    assert torch.allclose(prediction["x1"], torch.zeros_like(prediction["x1"]), atol=1e-6)

    true_inner_prods = two_variable_sample["x1"] * random_features[:, 0]
    transformed_inner_prods = torch.cos(true_inner_prods - (math.pi / 2)) * coeff_alpha
    true_x2_prediction = math.sqrt(2 / 5) * torch.sum(transformed_inner_prods, dim=-1)
    true_x2_prediction = true_x2_prediction.unsqueeze(-1).repeat(1, 2)
    assert torch.allclose(prediction["x2"], true_x2_prediction)


def test_non_linear_forward_multigraph(two_variable_dict, two_variable_graphs, two_variable_sample):
    random_features = torch.rand((5, 3))
    coeff_alpha = torch.rand((5,))
    func_rel = RFFFunctionalRelationships(two_variable_dict, random_features, coeff_alpha)
    prediction = func_rel(two_variable_sample.unsqueeze(1).expand(3, 2), two_variable_graphs)

    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 2, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2, 2)

    # x1 and x2 are initial nodes for graphs 0 and 1 respectively
    assert torch.allclose(prediction["x1"][:, 0, :], torch.zeros_like(prediction["x1"][:, 0, :]), atol=1e-6)
    assert torch.allclose(prediction["x2"][:, 1, :], torch.zeros_like(prediction["x2"][:, 1, :]), atol=1e-6)

    true_inner_prods = two_variable_sample["x1"] * random_features[:, 0]
    transformed_inner_prods = torch.cos(true_inner_prods - (math.pi / 2)) * coeff_alpha
    true_x2_prediction = math.sqrt(2 / 5) * torch.sum(transformed_inner_prods, dim=-1)
    true_x2_prediction = true_x2_prediction.unsqueeze(-1).repeat(1, 2)
    assert torch.allclose(prediction["x2"][:, 0, :], true_x2_prediction)

    true_inner_prods = torch.matmul(two_variable_sample["x2"], random_features[:, 1:].transpose(-2, -1))
    transformed_inner_prods = torch.cos(true_inner_prods - (math.pi / 2)) * coeff_alpha
    true_x1_prediction = math.sqrt(2 / 5) * torch.sum(transformed_inner_prods, dim=-1)
    true_x1_prediction = true_x1_prediction.unsqueeze(-1)
    assert torch.allclose(prediction["x1"][:, 1, :], true_x1_prediction)


def test_non_linear_forward_full(two_variable_dict, two_variable_graph, two_variable_sample):
    length_scales = torch.rand((3,))
    out_scales = torch.rand((3,))
    random_features = torch.rand((5, 3))
    coeff_alpha = torch.rand((5,))
    func_rel = RFFFunctionalRelationships(
        two_variable_dict,
        random_features,
        coeff_alpha,
        initial_length_scales=length_scales,
        initial_output_scales=out_scales,
    )
    prediction = func_rel(two_variable_sample, two_variable_graph)
    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2)
    assert torch.allclose(prediction["x1"], torch.zeros_like(prediction["x1"]), atol=1e-6)

    true_inner_prods = two_variable_sample["x1"] * random_features[:, 0]
    true_inner_prods_rescaled = true_inner_prods.repeat(2, 1, 1) / length_scales[1:].reshape(2, 1, 1)
    transformed_inner_prods = torch.cos(true_inner_prods_rescaled - (math.pi / 2)) * coeff_alpha
    true_x2_prediction = math.sqrt(2 / 5) * out_scales[1:].reshape(2, 1) * torch.sum(transformed_inner_prods, dim=-1)
    assert torch.allclose(prediction["x2"], true_x2_prediction.transpose(-1, -2))


def test_non_linear_forward_multigraph_full(two_variable_dict, two_variable_graphs, two_variable_sample):
    length_scales = torch.rand((3,))
    out_scales = torch.rand((3,))
    random_features = torch.rand((5, 3))
    coeff_alpha = torch.rand((5,))
    func_rel = RFFFunctionalRelationships(
        two_variable_dict,
        random_features,
        coeff_alpha,
        initial_length_scales=length_scales,
        initial_output_scales=out_scales,
    )
    prediction = func_rel(two_variable_sample.unsqueeze(1).expand(3, 2), two_variable_graphs)

    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 2, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2, 2)

    # x1 and x2 are initial nodes for graphs 0 and 1 respectively
    assert torch.allclose(prediction["x1"][:, 0, :], torch.zeros_like(prediction["x1"][:, 0, :]), atol=1e-6)
    assert torch.allclose(prediction["x2"][:, 1, :], torch.zeros_like(prediction["x2"][:, 1, :]), atol=1e-6)

    true_inner_prods = two_variable_sample["x1"] * random_features[:, 0]
    true_inner_prods_rescaled = true_inner_prods.repeat(2, 1, 1) / length_scales[1:].reshape(2, 1, 1)
    transformed_inner_prods = torch.cos(true_inner_prods_rescaled - (math.pi / 2)) * coeff_alpha
    true_x2_prediction = math.sqrt(2 / 5) * out_scales[1:].reshape(2, 1) * torch.sum(transformed_inner_prods, dim=-1)
    assert torch.allclose(prediction["x2"][:, 0, :], true_x2_prediction.transpose(-1, -2))

    true_inner_prods = torch.matmul(two_variable_sample["x2"], random_features[:, 1:].transpose(-2, -1))
    true_inner_prods_rescaled = true_inner_prods / length_scales[0]
    transformed_inner_prods = torch.cos(true_inner_prods_rescaled - (math.pi / 2)) * coeff_alpha
    true_x1_prediction = math.sqrt(2 / 5) * out_scales[0] * torch.sum(transformed_inner_prods, dim=-1)
    true_x1_prediction = true_x1_prediction.unsqueeze(-1)
    assert torch.allclose(prediction["x1"][:, 1, :], true_x1_prediction)


def test_non_linear_forward_full_with_bias_and_angle(two_variable_dict, two_variable_graph, two_variable_sample):
    length_scales = torch.rand((3,))
    out_scales = torch.rand((3,))
    random_features = torch.rand((5, 3))
    coeff_alpha = torch.rand((5,))
    initial_bias = torch.rand((3,))
    initial_angles = torch.rand((5,))
    func_rel = RFFFunctionalRelationships(
        two_variable_dict,
        random_features,
        coeff_alpha,
        initial_bias=initial_bias,
        initial_length_scales=length_scales,
        initial_output_scales=out_scales,
        initial_angles=initial_angles,
    )
    prediction = func_rel(two_variable_sample, two_variable_graph)
    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2)
    res = initial_bias[0] + out_scales[0] * math.sqrt(2 / 5) * torch.sum(torch.cos(initial_angles) * coeff_alpha)
    assert torch.allclose(prediction["x1"], res.repeat(3, 1), atol=1e-6)

    true_inner_prods = two_variable_sample["x1"] * random_features[:, 0]
    true_inner_prods_rescaled = true_inner_prods.repeat(2, 1, 1) / length_scales[1:].reshape(2, 1, 1)
    transformed_inner_prods = torch.cos(true_inner_prods_rescaled + initial_angles) * coeff_alpha
    true_x2_prediction = initial_bias[1:].reshape(2, 1) + math.sqrt(2 / 5) * out_scales[1:].reshape(2, 1) * torch.sum(
        transformed_inner_prods, dim=-1
    )
    assert torch.allclose(prediction["x2"], true_x2_prediction.transpose(-1, -2))


def test_non_linear_forward_multigraph_full_with_bias_and_angle(
    two_variable_dict, two_variable_graphs, two_variable_sample
):
    length_scales = torch.rand((3,))
    out_scales = torch.rand((3,))
    random_features = torch.rand((5, 3))
    coeff_alpha = torch.rand((5,))
    initial_bias = torch.rand((3,))
    initial_angles = torch.rand((5,))
    func_rel = RFFFunctionalRelationships(
        two_variable_dict,
        random_features,
        coeff_alpha,
        initial_bias=initial_bias,
        initial_length_scales=length_scales,
        initial_output_scales=out_scales,
        initial_angles=initial_angles,
    )
    prediction = func_rel(two_variable_sample.unsqueeze(1).expand(3, 2), two_variable_graphs)

    assert set(prediction.keys()) == {"x1", "x2"}
    assert prediction["x1"].shape == (3, 2, 1), f"got {prediction['x1'].shape}"
    assert prediction["x2"].shape == (3, 2, 2)

    # x1 and x2 are initial nodes for graphs 0 and 1 respectively
    res_1 = initial_bias[0] + out_scales[0] * math.sqrt(2 / 5) * torch.sum(torch.cos(initial_angles) * coeff_alpha)
    assert torch.allclose(prediction["x1"][:, 0, :], res_1, atol=1e-6)
    res_2 = initial_bias[1:] + out_scales[1:] * math.sqrt(2 / 5) * torch.sum(torch.cos(initial_angles) * coeff_alpha)
    assert torch.allclose(prediction["x2"][:, 1, :], res_2.repeat(3, 1), atol=1e-6)

    true_inner_prods = two_variable_sample["x1"] * random_features[:, 0]
    true_inner_prods_rescaled = true_inner_prods.repeat(2, 1, 1) / length_scales[1:].reshape(2, 1, 1)
    transformed_inner_prods = torch.cos(true_inner_prods_rescaled + initial_angles) * coeff_alpha
    true_x2_prediction = initial_bias[1:].reshape(2, 1) + math.sqrt(2 / 5) * out_scales[1:].reshape(2, 1) * torch.sum(
        transformed_inner_prods, dim=-1
    )
    assert torch.allclose(prediction["x2"][:, 0, :], true_x2_prediction.transpose(-1, -2))

    true_inner_prods = torch.matmul(two_variable_sample["x2"], random_features[:, 1:].transpose(-2, -1))
    true_inner_prods_rescaled = true_inner_prods / length_scales[0]
    transformed_inner_prods = torch.cos(true_inner_prods_rescaled + initial_angles) * coeff_alpha
    true_x1_prediction = initial_bias[0] + math.sqrt(2 / 5) * out_scales[0] * torch.sum(transformed_inner_prods, dim=-1)
    true_x1_prediction = true_x1_prediction.unsqueeze(-1)
    assert torch.allclose(prediction["x1"][:, 1, :], true_x1_prediction)
