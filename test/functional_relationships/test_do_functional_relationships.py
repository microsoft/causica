import pytest
import torch
from tensordict import TensorDict

from causica.functional_relationships import LinearFunctionalRelationships, create_do_functional_relationship


@pytest.fixture(name="two_variable_dict")
def fixture_two_variable_dict():
    return {"x1": torch.Size([1]), "x2": torch.Size([2])}


@pytest.fixture(name="three_variable_dict")
def fixture_three_variable_dict():
    return {"x1": torch.Size([2]), "x2": torch.Size([2]), "x3": torch.Size([1])}


def test_do_linear_graph_stack(two_variable_dict):
    graph = torch.tensor([[[0, 1], [0, 0.0]], [[0, 0], [1, 0.0]]])
    coef_matrix = torch.randn((3, 3))
    interventions = TensorDict({"x2": torch.tensor([1.42, 0.42])}, batch_size=torch.Size())
    func = LinearFunctionalRelationships(two_variable_dict, coef_matrix)
    do_func, do_graph = create_do_functional_relationship(interventions, func, graph)
    array = torch.linspace(-2, 2, 100)[..., None, None].expand(-1, 2, 1)
    prediction = do_func.forward(TensorDict({"x1": array}, batch_size=array.shape[:-1]), graphs=do_graph)
    assert prediction["x1"].shape == (100, 2, 1)
    assert torch.allclose(prediction["x1"][:, 0, :], torch.tensor(0.0))

    true_pred = torch.matmul(torch.tensor([1.42, 0.42]).unsqueeze(-2), coef_matrix[1:, :1]).squeeze()
    assert torch.allclose(prediction["x1"][:, 1, :], true_pred)


def test_linear_3d_graph_do_1_node(three_variable_dict):
    coef_matrix = torch.rand((5, 5))
    graph = torch.zeros(3, 3)
    graph[0, 1] = graph[0, 2] = 1
    func = LinearFunctionalRelationships(three_variable_dict, coef_matrix)
    interventions = TensorDict({"x1": torch.tensor([1.42, 0.42])}, batch_size=torch.Size())
    do_func, do_graph = create_do_functional_relationship(interventions, func, graph)
    test_val = torch.rand(100, 3)

    input_noise = TensorDict({"x2": test_val[:, 0:2], "x3": test_val[:, 2:]}, batch_size=test_val.shape[:-1])
    prediction = do_func.forward(input_noise, graphs=do_graph)
    assert "x1" not in input_noise.keys()
    assert prediction["x2"].shape == (100, 2)
    assert prediction["x3"].shape == (100, 1)

    true_pred = torch.matmul(torch.tensor([1.42, 0.42]).unsqueeze(-2), coef_matrix[:2, 2:]).squeeze()
    assert torch.allclose(prediction["x2"], true_pred[:2])
    assert torch.allclose(prediction["x3"], true_pred[2:])


def test_linear_3d_graph_do_2_nodes(three_variable_dict):
    coef_matrix = torch.rand((5, 5))
    graph = torch.zeros(3, 3)
    graph[0, 1] = graph[0, 2] = graph[1, 2] = 1
    func = LinearFunctionalRelationships(three_variable_dict, coef_matrix)
    interventions = TensorDict(
        {"x1": torch.tensor([1.42, 0.42]), "x2": torch.tensor([-0.42, 0.402])}, batch_size=torch.Size()
    )
    do_func, do_graph = create_do_functional_relationship(interventions, func, graph)

    test_val = torch.linspace(-2, 2, 100).unsqueeze(-1)

    input_noise = TensorDict({"x3": test_val}, batch_size=test_val.shape[:-1])
    prediction = do_func(input_noise, do_graph)
    assert "x1" not in input_noise.keys() and "x2" not in input_noise.keys()
    assert prediction["x3"].shape == (100, 1)

    true_pred = torch.matmul(torch.tensor([1.42, 0.42, -0.42, 0.402]).unsqueeze(-2), coef_matrix[:4, 4:]).squeeze()
    assert torch.allclose(true_pred, prediction["x3"])
