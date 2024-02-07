import pytest
import torch
import torch.distributions as td
from tensordict import TensorDict

from causica.datasets.variable_types import VariableTypeEnum
from causica.distributions import JointNoiseModule, create_noise_modules
from causica.distributions.noise.joint import ContinuousNoiseDist
from causica.functional_relationships import LinearFunctionalRelationships
from causica.sem.distribution_parameters_sem import DistributionParametersSEM

from . import create_lingauss_sem, create_rffgauss_sem


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

    # Test multi-dim interventions
    do_sem = sem.do(
        TensorDict(
            {"x2": intervention_value.expand(3, 2)},
            batch_size=[
                3,
            ],
        )
    )
    noise = sem.sample_noise((10, 3, 1))
    do_sample = do_sem.noise_to_sample(noise)
    sample = sem.noise_to_sample(noise)
    if graph[1, 0] == 1.0:
        assert torch.allclose(do_sample["x1"], expected_mean + noise["x1"])
    else:
        assert torch.allclose(do_sample["x1"], sample["x1"])


@pytest.mark.parametrize("graph", [torch.tensor([[0, 0], [1, 0.0]]), torch.tensor([[0, 1], [0, 0.0]])])
@pytest.mark.parametrize(
    "intervention_variable,intervention_value",
    [("x1", [[1.42], [0.42], [1.12]]), ("x2", [[1.42, 0.42], [0.42, 1.42], [0.42, 0.42]])],
)
def test_batched_intervention_2d_graph(graph, intervention_variable, intervention_value, two_variable_dict):
    rff_features = torch.rand((10, 3))
    coef_matrix = torch.rand((10,))
    sem = create_rffgauss_sem(two_variable_dict, rff_features, coef_matrix, graph)
    variable_names = set(two_variable_dict.keys())
    sampled_variables = variable_names - {intervention_variable}
    assert len(sampled_variables) == 1
    sampled_variable = sampled_variables.pop()
    intervention_value = torch.tensor(intervention_value)
    batched_do_sem = sem.do(
        TensorDict(
            {intervention_variable: intervention_value},
            batch_size=[
                3,
            ],
        )
    )
    noise = batched_do_sem.sample_noise((10,))
    do_sample = batched_do_sem.noise_to_sample(noise)

    non_batch_sample_list = []
    for i, intervention in enumerate(intervention_value):
        non_batch_sample_list.append(
            sem.do(TensorDict({intervention_variable: intervention}, batch_size=tuple()))
            .noise_to_sample(noise[:, i, None])
            .squeeze(1)
        )
    non_batch_sample = torch.stack(non_batch_sample_list, dim=1)
    diff = torch.abs(do_sample[sampled_variable] - non_batch_sample[sampled_variable])
    assert torch.allclose(
        do_sample[sampled_variable], non_batch_sample[sampled_variable], atol=1e-5, rtol=1e-4
    ), f"{diff.max()}"


def test_batched_intervention_3d_graph(three_variable_dict):
    graph = torch.zeros(3, 3)
    graph[0, 1] = graph[0, 2] = graph[1, 2] = 1
    rff_features = torch.rand((10, 5))
    coef_matrix = torch.rand((10,))
    sem = create_rffgauss_sem(three_variable_dict, rff_features, coef_matrix, graph)
    intervention_value = torch.tensor([[1.42, 0.42], [0.42, 1.42], [0.42, 0.42]])
    batched_do_sem = sem.do(
        TensorDict(
            {"x2": intervention_value},
            batch_size=[
                3,
            ],
        )
    )
    noise = batched_do_sem.sample_noise((10,))
    do_sample = batched_do_sem.noise_to_sample(noise)
    inferred_noise = batched_do_sem.sample_to_noise(do_sample)
    assert all(torch.allclose(noise[key], inferred_noise[key]) for key in noise.keys() if key != "x2")

    non_batch_sample_list = []
    for i, intervention in enumerate(intervention_value):
        non_batch_sample_list.append(
            sem.do(TensorDict({"x2": intervention}, batch_size=tuple())).noise_to_sample(noise[:, i, None]).squeeze(1)
        )

    non_batch_sample = torch.stack(non_batch_sample_list, dim=1)

    assert torch.allclose(do_sample["x1"], non_batch_sample["x1"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(do_sample["x3"], non_batch_sample["x3"], atol=1e-5, rtol=1e-4)


def test_intervention_batched_3d_graph(three_variable_dict):
    graph = torch.zeros(3, 3)
    graph[0, 1] = graph[0, 2] = graph[1, 2] = 1
    graphs = torch.stack([graph, graph], dim=0)
    rff_features = torch.rand((10, 5))
    coef_matrix = torch.rand((10,))
    sem = create_rffgauss_sem(three_variable_dict, rff_features, coef_matrix, graphs)

    regular_noise = sem.sample_noise(torch.Size([10]))
    samples = sem.noise_to_sample(regular_noise)
    inferred_noise = sem.sample_to_noise(samples)
    assert all(
        torch.allclose(regular_noise[key], inferred_noise[key], atol=1e-5, rtol=1e-4) for key in regular_noise.keys()
    )

    intervention_value = torch.tensor([1.42, 0.42])
    do_sem = sem.do(
        TensorDict(
            {"x2": intervention_value},
            batch_size=torch.Size([]),
        )
    )
    noise = do_sem.sample_noise(torch.Size([10]))
    do_sample = do_sem.noise_to_sample(noise)
    inferred_noise = do_sem.sample_to_noise(do_sample)
    assert all(torch.allclose(noise[key], inferred_noise[key]) for key in noise.keys() if key != "x2")


def test_batched_intervention_batched_3d_graph(three_variable_dict):
    graph = torch.zeros(3, 3)
    graph[0, 1] = graph[0, 2] = graph[1, 2] = 1
    graphs = torch.stack([graph, graph], dim=0)
    rff_features = torch.rand((10, 5))
    coef_matrix = torch.rand((10,))
    sem = create_rffgauss_sem(three_variable_dict, rff_features, coef_matrix, graphs)
    intervention_value = torch.tensor([[1.42, 0.42], [0.42, 1.42], [0.42, 0.42]])
    batched_do_sem = sem.do(
        TensorDict(
            {"x2": intervention_value},
            batch_size=[
                3,
            ],
        )
    )
    noise = batched_do_sem.sample_noise((10,))
    do_sample = batched_do_sem.noise_to_sample(noise)
    assert do_sample.batch_size == torch.Size([10, 3, 2])  # 10 samples, 3 interventions, 2 graphs
    inferred_noise = batched_do_sem.sample_to_noise(do_sample)
    assert all(torch.allclose(noise[key], inferred_noise[key]) for key in noise.keys() if key != "x2")
    non_batch_sample_list = []
    for i, intervention in enumerate(intervention_value):
        non_batch_sample_list.append(
            sem.do(TensorDict({"x2": intervention}, batch_size=tuple())).noise_to_sample(noise[:, i, None]).squeeze(1)
        )
    non_batch_sample = torch.stack(non_batch_sample_list, dim=1)

    assert torch.allclose(do_sample["x1"], non_batch_sample["x1"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(do_sample["x3"], non_batch_sample["x3"], atol=1e-5, rtol=1e-4)


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


def test_linear_sem_3d_graph_condition_1_node(three_variable_dict):
    coef_matrix = torch.rand((5, 5))
    graph = torch.zeros(3, 3)
    graph[0, 1] = graph[0, 2] = graph[1, 2] = 1
    sem = create_lingauss_sem(three_variable_dict, coef_matrix, graph)
    condition_value = torch.tensor([1.42, 0.42])
    condition_sem = sem.condition(TensorDict({"x1": condition_value}, batch_size=tuple()))
    test_val = torch.rand(100, 3)
    log_probs = condition_sem.log_prob(TensorDict({"x2": test_val[:, 0:2], "x3": test_val[:, 2:]}, batch_size=[100]))

    with pytest.raises(NotImplementedError, match=r"on x2"):
        sem.condition(TensorDict({"x2": condition_value}, batch_size=tuple()))

    condition_sem = sem.condition(TensorDict({"x1": condition_value, "x2": condition_value}, batch_size=tuple()))
    test_val = torch.rand(100, 1)
    log_probs = condition_sem.log_prob(TensorDict({"x3": test_val}, batch_size=[100]))
    expected_mean = torch.einsum("i,ij->j", torch.cat([condition_value, condition_value]), coef_matrix[:4, 4:])
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
    noise_modules = create_noise_modules(
        shapes=two_variable_dict,
        types=dict.fromkeys(two_variable_dict, VariableTypeEnum.BINARY),
        continuous_noise_dist=ContinuousNoiseDist.GAUSSIAN,
    )
    noise_dist = JointNoiseModule(noise_modules)
    sem = DistributionParametersSEM(graph=graph, noise_dist=noise_dist, func=func)
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
    noise_modules = create_noise_modules(
        shapes=three_variable_dict,
        types=dict.fromkeys(three_variable_dict, VariableTypeEnum.BINARY),
        continuous_noise_dist=ContinuousNoiseDist.GAUSSIAN,
    )
    noise_dist = JointNoiseModule(noise_modules)
    sem = DistributionParametersSEM(graph=graph, noise_dist=noise_dist, func=func)
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
    noise_modules = create_noise_modules(
        shapes=three_variable_dict,
        types=dict.fromkeys(three_variable_dict, VariableTypeEnum.BINARY),
        continuous_noise_dist=ContinuousNoiseDist.GAUSSIAN,
    )
    noise_dist = JointNoiseModule(noise_modules)
    sem = DistributionParametersSEM(graph=graph, noise_dist=noise_dist, func=func)
    int_data = TensorDict({"x1": torch.tensor([1.42, 0.42]), "x2": torch.tensor([-0.42, 0.402])}, batch_size=tuple())
    do_sem = sem.do(int_data)
    array = torch.bernoulli(0.5 * torch.ones(100, 1))
    log_probs = do_sem.log_prob(TensorDict({"x3": array}, batch_size=array.shape[:-1]))
    expected_logits = torch.einsum("i,ij->j", torch.cat((int_data["x1"], int_data["x2"])), coef_matrix[:4, 4:])
    expected_log_probs = td.Independent(td.Bernoulli(logits=expected_logits), 1).log_prob(array)
    torch.testing.assert_close(log_probs, expected_log_probs)
