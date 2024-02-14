import os
import tempfile

import numpy as np
import pytest
import torch
import torch.distributions as td
from tensordict import TensorDict

from causica.data_generation.generate_data import (
    generate_sem_sampler,
    get_variable_definitions,
    plot_dataset,
    sample_counterfactual,
    sample_dataset,
    sample_intervention,
    sample_treatment_and_effect,
)
from causica.data_generation.samplers.functional_relationships_sampler import LinearRelationshipsSampler
from causica.data_generation.samplers.noise_dist_sampler import (
    BernoulliNoiseModuleSampler,
    CategoricalNoiseModuleSampler,
    JointNoiseModuleSampler,
    UnivariateNormalNoiseModuleSampler,
)
from causica.data_generation.samplers.sem_sampler import SEMSampler
from causica.datasets.causica_dataset_format import DataEnum, Variable, VariablesMetadata, save_data
from causica.distributions import ErdosRenyiDAGDistribution

try:
    import seaborn as sns
except ImportError:
    sns = None


@pytest.fixture(name="five_variable_one_dim_metadata")
def fixture_five_variable_one_dim_metadata() -> VariablesMetadata:
    return VariablesMetadata([Variable(f"x_{i}", f"x_{i}") for i in range(5)])


@pytest.fixture(name="five_variable_two_dim_metadata")
def fixture_five_variable_two_dim_metadata() -> VariablesMetadata:
    return VariablesMetadata([Variable(f"x_{i // 2}", f"x_{i}") for i in range(10)])


@pytest.fixture(name="five_node_graph")
def fixture_five_node_graph() -> torch.Tensor:
    graph = torch.zeros(5, 5)
    graph[0, 1] = 1
    graph[1, 2] = 1
    graph[2, 3] = 1

    return graph


@pytest.fixture(name="mixture_sem_sampler_and_shapes")
def fixture_mixture_sem_sampler_and_shapes() -> tuple[SEMSampler, dict[str, torch.Size]]:
    shapes_dict = {
        "x_0": torch.Size([1]),
        "x_1": torch.Size([5]),
        "x_2": torch.Size([1]),
        "x_3": torch.Size([3]),
        "x_4": torch.Size([4]),
    }

    noise_dist_samplers = {
        "x_0": UnivariateNormalNoiseModuleSampler(std_dist=td.Uniform(low=0.2, high=2.0), dim=1),
        "x_1": UnivariateNormalNoiseModuleSampler(std_dist=td.Uniform(low=0.2, high=2.0), dim=5),
        "x_2": BernoulliNoiseModuleSampler(base_logits_dist=td.Uniform(low=0.2, high=2.0), dim=1),
        "x_3": BernoulliNoiseModuleSampler(
            base_logits_dist=td.Uniform(low=0.2 * torch.ones([3]), high=2.0 * torch.ones([3])), dim=3
        ),
        "x_4": CategoricalNoiseModuleSampler(
            base_logits_dist=td.Uniform(low=0.2 * torch.ones([4]), high=2.0 * torch.ones([4])), num_classes=4
        ),
    }

    # Create adjacency distribution, joint noise module sampler, and functional relationships sampler
    adjacency_dist = ErdosRenyiDAGDistribution(num_nodes=5, num_edges=torch.tensor(10))
    joint_noise_module_sampler = JointNoiseModuleSampler(noise_dist_samplers)
    initial_linear_coefficient_matrix_shape = sum(shape[0] for shape in shapes_dict.values())
    functional_relationships_sampler = LinearRelationshipsSampler(
        td.Uniform(
            low=torch.ones((initial_linear_coefficient_matrix_shape, initial_linear_coefficient_matrix_shape)),
            high=3.0 * torch.ones((initial_linear_coefficient_matrix_shape, initial_linear_coefficient_matrix_shape)),
        ),
        shapes_dict,
    )

    return SEMSampler(adjacency_dist, joint_noise_module_sampler, functional_relationships_sampler), shapes_dict


def test_sample_interventions(mixture_sem_sampler_and_shapes: tuple[SEMSampler, dict[str, torch.Size]]):
    mixture_sem_sampler, shapes_dict = mixture_sem_sampler_and_shapes
    sem = mixture_sem_sampler.sample()

    noise = sem.sample_noise(torch.Size([5]))
    observations = sem.noise_to_sample(noise)
    num_intervention_samples = 3
    intervention = sample_intervention(sem, observations, num_intervention_samples, "x_1")

    for v, shape in shapes_dict.items():
        assert intervention.intervention_data[v].shape == torch.Size([num_intervention_samples]) + shape
    # test ordering of intervention data
    assert list(intervention.intervention_data.keys()) == list(shapes_dict.keys())

    assert all(val.ndim == 1 for val in intervention.intervention_values.values())
    assert intervention.intervention_values["x_1"].shape == shapes_dict["x_1"]
    assert all(intervention.intervention_values["x_1"] >= observations["x_1"].min())
    assert all(intervention.intervention_values["x_1"] <= observations["x_1"].max())


def test_sample_counterfactuals(mixture_sem_sampler_and_shapes: tuple[SEMSampler, dict[str, torch.Size]]):
    mixture_sem_sampler, shapes_dict = mixture_sem_sampler_and_shapes
    sem = mixture_sem_sampler.sample()

    num_samples = 5
    noise = sem.sample_noise(torch.Size([num_samples]))
    observations = sem.noise_to_sample(noise)

    counterfactual = sample_counterfactual(sem, observations, noise, "x_1")

    for v, shape in shapes_dict.items():
        assert counterfactual.counterfactual_data[v].shape == torch.Size([num_samples]) + shape
    # test ordering of intervention data
    assert list(counterfactual.counterfactual_data.keys()) == ["x_0", "x_1", "x_2", "x_3", "x_4"]

    assert all(val.ndim == 1 for val in counterfactual.intervention_values.values())
    assert counterfactual.intervention_values["x_1"].shape == shapes_dict["x_1"]
    assert all(counterfactual.intervention_values["x_1"] >= observations["x_1"].min())
    assert all(counterfactual.intervention_values["x_1"] <= observations["x_1"].max())

    torch.testing.assert_close(counterfactual.factual_data, observations)
    torch.testing.assert_close(
        sem.do(counterfactual.intervention_values).noise_to_sample(noise), counterfactual.counterfactual_data
    )


def test_generate_sem_er(five_variable_one_dim_metadata: VariablesMetadata):
    variables = get_variable_definitions(num_nodes=5)
    sem = generate_sem_sampler(
        variables=variables, graph_file=None, num_edges=5, graph_type="er", function_type="linear"
    ).sample()
    assert len(variables.variables) == 5
    assert variables == five_variable_one_dim_metadata

    samples = sem.sample(torch.Size([5]))
    assert samples["x_0"].shape == torch.Size([5, 1])
    assert samples["x_1"].shape == torch.Size([5, 1])
    assert samples["x_2"].shape == torch.Size([5, 1])
    assert samples["x_3"].shape == torch.Size([5, 1])
    assert samples["x_4"].shape == torch.Size([5, 1])


def test_generate_sem_variable_groups(five_variable_two_dim_metadata: VariablesMetadata):
    tmpdir = tempfile.mkdtemp()

    save_data(tmpdir, five_variable_two_dim_metadata, DataEnum.VARIABLES_JSON)
    variable_json_path = os.path.join(tmpdir, DataEnum.VARIABLES_JSON.value)
    variables = get_variable_definitions(variable_json_path=variable_json_path)
    sem = generate_sem_sampler(
        variables=variables,
        graph_file=None,
        num_edges=5,
        graph_type="er",
        function_type="linear",
    ).sample()
    assert len(variables.variables) == 10
    assert len(sem.node_names) == 5

    samples = sem.sample(torch.Size([5]))
    assert samples["x_0"].shape == torch.Size([5, 2])
    assert samples["x_1"].shape == torch.Size([5, 2])
    assert samples["x_2"].shape == torch.Size([5, 2])
    assert samples["x_3"].shape == torch.Size([5, 2])
    assert samples["x_4"].shape == torch.Size([5, 2])


def test_generate_sem_numpy(five_node_graph: torch.Tensor, five_variable_one_dim_metadata: VariablesMetadata):
    tmpfile = tempfile.mkstemp(suffix=".npy")[1]

    np.save(tmpfile, five_node_graph.numpy())
    assert os.path.exists(tmpfile)

    variables = get_variable_definitions(num_nodes=5)

    sem = generate_sem_sampler(
        variables=variables,
        graph_file=tmpfile,
        num_edges=None,
        graph_type="numpy",
        function_type="linear",
    ).sample()
    assert len(variables.variables) == 5
    assert variables == five_variable_one_dim_metadata

    samples = sem.sample(torch.Size([5]))
    assert samples["x_0"].shape == torch.Size([5, 1])
    assert samples["x_1"].shape == torch.Size([5, 1])
    assert samples["x_2"].shape == torch.Size([5, 1])
    assert samples["x_3"].shape == torch.Size([5, 1])
    assert samples["x_4"].shape == torch.Size([5, 1])

    torch.testing.assert_close(sem.graph, five_node_graph)


def test_sample_dataset(mixture_sem_sampler_and_shapes: tuple[SEMSampler, dict[str, torch.Size]]):
    mixture_sem_sampler, shapes_dict = mixture_sem_sampler_and_shapes
    sem = mixture_sem_sampler.sample()

    num_train = 5
    num_test = 3
    num_interventions = 2

    dataset = sample_dataset(
        sem=sem,
        sample_dataset_size=torch.Size([num_train + num_test * 2]),
        num_interventions=num_interventions,
        num_intervention_samples=num_test,
        sample_interventions=True,
        sample_counterfactuals=True,
    )

    assert dataset.noise is not None
    for k, v in shapes_dict.items():
        assert dataset.observations[k].shape == torch.Size([num_train + num_test * 2]) + v
        assert dataset.noise[k].shape == torch.Size([num_train + num_test * 2]) + v

    assert dataset.interventions is not None
    assert len(dataset.interventions) == num_interventions
    assert dataset.counterfactuals is not None
    assert len(dataset.counterfactuals) == num_interventions


def test_plot_data(five_variable_one_dim_metadata: VariablesMetadata):
    batch_size = 5
    data = TensorDict({f"x_{i}": torch.rand([batch_size, 1]) for i in range(5)}, batch_size=torch.Size([batch_size]))

    datadir = tempfile.mkdtemp()
    if sns is not None:
        plot_dataset(data, five_variable_one_dim_metadata, plot_kind="scatter", datadir=datadir)
        assert os.path.exists(os.path.join(datadir, "data_distribution.png"))
    else:
        with pytest.raises(ImportError):
            plot_dataset(data, five_variable_one_dim_metadata, plot_kind="scatter", datadir=datadir)


def test_plot_data_multi_dim(five_variable_two_dim_metadata: VariablesMetadata):
    batch_size = 5
    data = TensorDict({f"x_{i}": torch.rand([batch_size, 2]) for i in range(5)}, batch_size=torch.Size([batch_size]))

    datadir = tempfile.mkdtemp()
    if sns is not None:
        plot_dataset(data, five_variable_two_dim_metadata, plot_kind="scatter", datadir=datadir)
        assert os.path.exists(os.path.join(datadir, "data_distribution.png"))
    else:
        with pytest.raises(ImportError):
            plot_dataset(data, five_variable_two_dim_metadata, plot_kind="scatter", datadir=datadir)


def test_sample_treatment_and_effect():
    graph = torch.zeros(5, 5)
    node_names = [f"x_{i}" for i in range(5)]

    with pytest.warns(UserWarning, match="No edges found."):
        treatment, effects = sample_treatment_and_effect(graph, node_names)

    graph[0, 1] = 1
    graph[2, 3] = 1

    treatment, effects = sample_treatment_and_effect(graph, node_names)

    assert (treatment == "x_0" and effects == ["x_1"]) or (treatment == "x_2" and effects == ["x_3"])
