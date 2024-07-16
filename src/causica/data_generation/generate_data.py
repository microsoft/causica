import argparse
import os
import random
import warnings
from collections import defaultdict
from typing import Counter, Sequence

import fsspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.distributions as td
from tensordict import TensorDict

from causica.data_generation.samplers.functional_relationships_sampler import (
    FunctionalRelationshipsSampler,
    LinearRelationshipsSampler,
    RFFFunctionalRelationshipsSampler,
)
from causica.data_generation.samplers.noise_dist_sampler import (
    BernoulliNoiseModuleSampler,
    CategoricalNoiseModuleSampler,
    JointNoiseModuleSampler,
    NoiseModuleSampler,
    UnivariateNormalNoiseModuleSampler,
)
from causica.data_generation.samplers.sem_sampler import SEMSampler
from causica.datasets.causal_dataset import CausalDataset
from causica.datasets.causica_dataset_format import (
    CounterfactualWithEffects,
    InterventionWithEffects,
    Variable,
    VariablesMetadata,
    get_group_variable_names,
    save_dataset,
)
from causica.datasets.interventional_data import CounterfactualData, InterventionData
from causica.datasets.tensordict_utils import expand_tensordict_groups
from causica.datasets.variable_types import VariableTypeEnum
from causica.distributions.adjacency import AdjacencyDistribution, ErdosRenyiDAGDistribution, FixedAdjacencyDistribution
from causica.sem.structural_equation_model import SEM

try:
    import seaborn as sns
except ImportError:
    sns = None


def get_graph_distribution(
    graph_type: str,
    num_nodes: int | None = None,
    num_edges: int | None = None,
    probs: float | None = None,
    graph_file: str | None = None,
    **storage_options,
) -> AdjacencyDistribution:
    """Get a graph from a graph type.

    Args:
        graph_type: The type of graph to generate. Either "er" for Erdos-Renyi or "numpy" for a numpy graph.
        num_nodes: The number of nodes in the graph. Not used if using a numpy graph.
        num_edges: The number of edges in the graph. Not used if using a numpy graph.
        probs: A float of the probability that an edge exists between 2 nodes
        graph_file: The path to a graph file if using a numpy graph.
        storage_options: The storage options to pass to fsspec.

    Returns:
        The adjacency matrix of the graph.
    """
    if graph_type == "er":
        if num_nodes is None:
            raise ValueError("Must provide num_nodes for Erdos-Renyi graph.")
        num_edges_tensor = None
        if num_edges is not None:
            num_edges_tensor = torch.tensor(num_edges, dtype=torch.float32)
        probs_tensor = None
        if probs is not None:
            probs_tensor = torch.tensor(probs)
        return ErdosRenyiDAGDistribution(num_nodes=num_nodes, num_edges=num_edges_tensor, probs=probs_tensor)
    if graph_type == "numpy":
        if graph_file is None:
            raise ValueError("Must provide graph_file for numpy graph.")
        with fsspec.open(graph_file, **storage_options) as f:
            return FixedAdjacencyDistribution(torch.tensor(np.load(f)))

    raise ValueError("Unknown graph type.")


def get_functional_relationship_sampler(
    function_type: str, shapes_dict: dict[str, torch.Size]
) -> FunctionalRelationshipsSampler:
    """Get a functional relationship from a function type.

    Args:
        function_type: The type of function to generate. Either "linear" or "rff".
        shapes_dict: A dictionary mapping variable names to their shapes.

    Returns:
        The functional relationship.
    """
    dim = sum(s.numel() for s in shapes_dict.values())
    if function_type == "linear":
        ones_matrix = torch.ones((dim, dim), dtype=torch.float32)
        functional_relationships_sampler: FunctionalRelationshipsSampler = LinearRelationshipsSampler(
            scale_dist=td.Uniform(
                low=ones_matrix,
                high=3.0 * ones_matrix,
            ),
            shapes_dict=shapes_dict,
        )
    elif function_type == "rff":
        num_rf = 100
        ones_matrix = torch.ones((num_rf, dim), dtype=torch.float32)
        ones = torch.ones((num_rf,), dtype=torch.float32)
        functional_relationships_sampler = RFFFunctionalRelationshipsSampler(
            rf_dist=td.Uniform(
                low=7.0 * ones_matrix,
                high=10.0 * ones_matrix,
            ),
            coeff_dist=td.Uniform(
                low=10.0 * ones,
                high=20.0 * ones,
            ),
            shapes_dict=shapes_dict,
        )
    else:
        raise ValueError(f"Unknown function type. Got {function_type}")

    return functional_relationships_sampler


def get_noise_module_sampler(variables: VariablesMetadata) -> JointNoiseModuleSampler:
    """Get a noise module sampler from a list of variables.

    Args:
        variables: Variable specifications to generate noise modules for.

    Returns:
        The noise module sampler.
    """
    size_dict = get_size_dict(variables)
    variable_type_dict = get_variable_type_dict(variables)

    noise_dist_samplers: dict[str, NoiseModuleSampler] = {}
    for group, size in size_dict.items():
        if variable_type_dict[group] == VariableTypeEnum.BINARY:
            noise_dist_samplers[group] = BernoulliNoiseModuleSampler(
                base_logits_dist=td.Uniform(low=-5, high=5), dim=size
            )
        elif variable_type_dict[group] == VariableTypeEnum.CATEGORICAL:
            noise_dist_samplers[group] = CategoricalNoiseModuleSampler(
                base_logits_dist=td.Uniform(low=-5, high=5),
                num_classes=size,
            )
        else:
            noise_dist_samplers[group] = UnivariateNormalNoiseModuleSampler(
                std_dist=td.Uniform(low=0.2, high=2.0), dim=size
            )

    return JointNoiseModuleSampler(noise_dist_samplers)


def get_variable_type_dict(
    variables: VariablesMetadata,
) -> dict[str, VariableTypeEnum]:
    """Get a dictionary mapping variable groups to their types.

    This also ensures that every variable in a group has the same type.

    Args:
        variables: Variable specifications to get the types for.

    Returns:
        The variable type dictionary.
    """
    variable_type_dict = defaultdict(list)
    for variable in variables.variables:
        variable_type_dict[variable.group_name].append(variable.type)

    for group, types in variable_type_dict.items():
        assert len(set(types)) == 1, f"Found multiple variable types in group {group}"

        if types[0] == VariableTypeEnum.CATEGORICAL:
            assert len(types) == 1, "Groups of categorical variables must have only one variable."

    return {k: v[0] for k, v in variable_type_dict.items()}


def get_size_dict(variables: VariablesMetadata) -> dict[str, int]:
    """Get a dictionary mapping variable groups to their sizes.

    Args:
        variables: Variable specifications to get the sizes for.

    Returns:
        The size dictionary.
    """
    size_dict = Counter(d.group_name for d in variables.variables)
    for variable in variables.variables:
        if variable.type == VariableTypeEnum.CATEGORICAL:
            assert variable.lower is not None and variable.upper is not None
            difference = variable.upper - variable.lower + 1
            assert difference // 1 == difference, ""
            assert size_dict[variable.group_name] == 1
            size_dict[variable.group_name] = int(difference)

    return size_dict


def sample_treatment_given_effects(graph: torch.Tensor, node_names: Sequence[str], effect_variables: list[str]) -> str:
    """Sample a treatment from the graph.

    Args:
        graph: The adjacency matrix of the graph.
        node_names: The names of the nodes in the graph.
        effect_variables: pre-specified effect variables.

    Returns:
        The treatment variable.
    """
    nx_graph = nx.from_numpy_array(graph.numpy(), create_using=nx.DiGraph)
    nx_graph = nx.relabel_nodes(nx_graph, dict(enumerate(node_names)))

    common_ancestors = nx.ancestors(nx_graph, effect_variables[0])
    for effect_variable in effect_variables[1:]:
        common_ancestors &= nx.ancestors(nx_graph, effect_variable)
    if common_ancestors:
        treatment = np.random.choice(list(common_ancestors), size=1, replace=False).item()
        return treatment
    warnings.warn("No common ancestors found for the given effects. Returning random sampled treatment.")
    possible_treatments = [node for node in node_names if node not in effect_variables]
    treatment = np.random.choice(possible_treatments, size=1, replace=False).item()
    return treatment


def sample_effects_given_treatment(
    graph: torch.Tensor,
    node_names: Sequence[str],
    treatment_variable: str,
    num_effects: int = 1,
) -> list[str]:
    """Sample effect variables from the graph given a treatment.

    Args:
        graph: The adjacency matrix of the graph.
        node_names: The names of the nodes in the graph.
        treatment_variable: The treatment variable from which to find effects.
        num_effects: The number of effect nodes to sample.

    Returns:
        The list of effect variables.
    """
    nx_graph = nx.from_numpy_array(graph.numpy(), create_using=nx.DiGraph)
    nx_graph = nx.relabel_nodes(nx_graph, dict(enumerate(node_names)))

    descendants = list(nx.descendants(nx_graph, treatment_variable))
    if not descendants:
        warnings.warn("No descendants found for the given treatment. Defaulting to random sampling.")
        possible_effects = [node for node in node_names if node != treatment_variable]
        return np.random.choice(possible_effects, size=num_effects, replace=False).tolist()

    return np.random.choice(descendants, size=num_effects, replace=False).tolist()


def sample_treatment_and_effect(
    graph: torch.Tensor,
    node_names: Sequence[str],
    ensure_effect: bool = True,
    num_effects: int = 1,
) -> tuple[str, list[str]]:
    """Sample a treatment and effects from a graph.

    Args:
        graph: The adjacency matrix of the graph.
        node_names: The names of the nodes in the graph.
        ensure_effect: Whether to ensure that there is a path from the treatment to the effect.
        num_effects: The number of effect nodes to sample.
    Returns:
        A tuple containing the treatment variable and a list of effect variables.
    """

    if ensure_effect:
        nx_graph = nx.from_numpy_array(graph.numpy(), create_using=nx.DiGraph)
        nx_graph = nx.relabel_nodes(nx_graph, dict(enumerate(node_names)))
        nodes_with_outgoing_edges = [n for n in node_names if any(True for _ in nx_graph.successors(n))]
        if nodes_with_outgoing_edges:
            treatment = np.random.choice(nodes_with_outgoing_edges, size=1, replace=False).item()
            descendants = list(nx.descendants(nx_graph, treatment))
            effects = np.random.choice(descendants, size=num_effects, replace=False).tolist()
            return treatment, effects
        warnings.warn("No nodes with descendants found in the graph. Defaulting to random sampling.")

    samples = np.random.choice(node_names, size=1 + num_effects, replace=False)
    treatment = samples[0]
    effects = samples[1:].tolist()

    return treatment, effects


def sample_dataset(
    sem: SEM,
    sample_dataset_size: torch.Size,
    num_interventions: int = 0,
    num_intervention_samples: int = 1000,
    sample_interventions: bool = False,
    sample_counterfactuals: bool = False,
    treatment_variable: str | None = None,
    effect_variables: list[str] | None = None,
) -> CausalDataset:
    """Sample a new dataset and returns it as a CausalDataset object.

    Args:
        sem: The SEM to sample from.
        sample_dataset_size: The size of the dataset to sample from the SEM
        num_interventions: The number of interventions to sample per dataset. If 0, no interventions are sampled.
        num_intervention_samples: The number of interventional samples to sample.
        sample_interventions: Whether to sample interventions.
        sample_counterfactuals: Whether to sample counterfactuals.
        treatment_variable: pre-specified treatment
        effet_variables: pre-specified effects
    Returns:
        A CausalDataset object holding the data, graph and potential interventions and counterfactuals.
    """

    noise = sem.sample_noise(sample_dataset_size)
    observations = sem.noise_to_sample(noise)

    interventions: list[InterventionWithEffects] = []
    counterfactuals: list[CounterfactualWithEffects] = []

    for _ in range(num_interventions):
        if treatment_variable and effect_variables:
            treatment, effects = treatment_variable, effect_variables
        elif treatment_variable:
            treatment, effects = treatment_variable, sample_effects_given_treatment(
                sem.graph, sem.node_names, treatment_variable
            )
        elif effect_variables:
            treatment, effects = (
                sample_treatment_given_effects(sem.graph, sem.node_names, effect_variables),
                effect_variables,
            )
        else:
            treatment, effects = sample_treatment_and_effect(sem.graph, sem.node_names)

        if sample_interventions:
            interventions.append(
                (
                    sample_intervention(sem, observations, num_intervention_samples, treatment=treatment),
                    sample_intervention(sem, observations, num_intervention_samples, treatment=treatment),
                    set(effects),
                )
            )

        if sample_counterfactuals:
            cf_noise = sem.sample_noise(
                torch.Size(
                    [
                        num_intervention_samples,
                    ]
                )
            )
            cf_observations = sem.noise_to_sample(cf_noise)

            counterfactuals.append(
                (
                    sample_counterfactual(sem, cf_observations, cf_noise, treatment=treatment),
                    sample_counterfactual(sem, cf_observations, cf_noise, treatment=treatment),
                    set(effects),
                )
            )

    return CausalDataset(
        observations=observations,
        noise=noise,
        graph=sem.graph,
        interventions=interventions,
        counterfactuals=counterfactuals,
    )


def sample_intervention_dict(tensordict_data: TensorDict, treatment: str | None = None) -> TensorDict:
    """Sample an intervention from a given SEM.

    This samples a random value for the treatment variable from the data. The value is sampled uniformly from the
    range of the treatment variable in the data.

    The treatment variable is chosen randomly across all nodes if not specified.

    Args:
        tensordict_data: Base data for sampling an intervention value.
        treatment: The name of the treatment variable. If None, a random variable is chosen across the tensordict keys.

    Returns:
        A TensorDict holding the intervention value.
    """
    if treatment is None:
        treatment = random.choice(list(tensordict_data.keys()))

    batch_axes = tuple(range(tensordict_data.batch_dims))
    treatment_shape = tensordict_data[treatment].shape[tensordict_data.batch_dims :]
    treatment_max = torch.amax(tensordict_data[treatment], dim=batch_axes)
    treatment_min = torch.amin(tensordict_data[treatment], dim=batch_axes)

    treatment_a = torch.rand(treatment_shape) * (treatment_max - treatment_min) + treatment_min

    return TensorDict({treatment: treatment_a}, batch_size=torch.Size())


def sample_intervention(
    sem: SEM, tensordict_data: TensorDict, num_intervention_samples: int, treatment: str | None = None
) -> InterventionData:
    """Sample an intervention and it's sample mean from a given SEM.

    Args:
        sem: SEM to sample interventional data from.
        tensordict_data: Base data for sampling an intervention value.
        num_intervention_samples: The number of samples to draw from the interventional distribution.
        treatment: The name of the treatment variable. If None, a random variable is chosen.

    Returns:
        an intervention data object
    """
    intervention_a = sample_intervention_dict(tensordict_data, treatment=treatment)

    intervention_a_samples = sem.do(intervention_a).sample(
        torch.Size(
            [
                num_intervention_samples,
            ]
        )
    )

    return InterventionData(
        intervention_a_samples,
        intervention_a,
        TensorDict({}, batch_size=torch.Size()),
    )


def sample_counterfactual(
    sem: SEM, factual_data: TensorDict, noise: TensorDict, treatment: str | None = None
) -> CounterfactualData:
    """Sample an intervention and it's sample mean from a given SEM.

    Args:
        sem: SEM to sample counterfactual data from.
        factual_data: Base data for sampling an counterfactual value.
        noise: Base noise for sampling an counterfactual value.
        treatment: The name of the treatment variable. If None, a random variable is chosen.

    Returns:
        an counterfactual data object
    """
    intervention_a = sample_intervention_dict(factual_data, treatment=treatment)

    counterfactuals = sem.do(intervention_a).noise_to_sample(noise)

    return CounterfactualData(counterfactuals, intervention_a, factual_data)


def get_variable_definitions(
    variable_json_path: str | None = None,
    num_nodes: int | None = None,
    storage_options: dict[str, str] | None = None,
) -> VariablesMetadata:
    """Get the variable definitions.

    Args:
        variable_json_path: The path to a json file containing the variables. This can be any fsspec compatible url.
        num_nodes: The number of nodes in the graph. Not used if using a numpy graph.
        storage_options: The storage options to pass to fsspec.

    Returns:
        The variables.
    """
    if storage_options is None:
        storage_options = {}

    if variable_json_path:
        with fsspec.open(variable_json_path, **storage_options) as f:
            variables = VariablesMetadata.from_json(f.read())  # type: ignore
    elif num_nodes is not None:
        variables = VariablesMetadata([Variable(f"x_{i}", f"x_{i}") for i in range(num_nodes)])
    else:
        raise ValueError("Must provide either variable_json_path or num_nodes.")

    return variables


def generate_sem_sampler(
    variables: VariablesMetadata,
    graph_file: str | None,
    num_edges: int | None,
    graph_type: str,
    function_type: str,
    probs: float | None = None,
    storage_options: dict[str, str] | None = None,
) -> SEMSampler:
    """Generates a SEM according to specifications

    Args:
        variable_json_path: The path to a json file containing the variables. This can be any fsspec compatible url.
        num_nodes: The number of nodes in the graph. Not used if using a numpy graph.
        graph_file: The path to a graph file if using a numpy graph. This can be any fsspec compatible url.
        num_edges: The number of edges in the graph. Not used if using a numpy graph.
        probs: A float of the probability that an edge exists between 2 nodes
        graph_type: The type of graph to generate. Either "er" for Erdos-Renyi or "numpy" for a numpy graph.
        function_type: The type of function to generate. Either "linear" or "rff".
        storage_options: The storage options to pass to fsspec.

    Returns:
        The variables and SEM.
    """
    if storage_options is None:
        storage_options = {}

    size_dict = get_size_dict(variables)
    shapes_dict = {k: torch.Size((s,)) for k, s in size_dict.items()}
    num_nodes = len(size_dict)

    print("Generating SEM sampler...")
    adjacency_dist = get_graph_distribution(graph_type, num_nodes, num_edges, probs, graph_file, **storage_options)
    functional_relationships_sampler = get_functional_relationship_sampler(function_type, shapes_dict)
    joint_noise_dist_sampler = get_noise_module_sampler(variables)

    sem_sampler = SEMSampler(
        adjacency_dist=adjacency_dist,
        joint_noise_module_sampler=joint_noise_dist_sampler,
        functional_relationships_sampler=functional_relationships_sampler,
    )

    return sem_sampler


def plot_dataset(
    data: TensorDict,
    variables: VariablesMetadata,
    plot_kind: str = "kde",
    plot_num: int = 10,
    datadir: str = "",
    storage_options: dict[str, str] | None = None,
) -> None:
    """Plot the data distribution.

    Args:
        data: The data to plot.
        variables: Variable specifications.
        plot_kind: Type of joint plot to create.
        plot_num: Maximum number of variables to plot.
        datadir: The directory to save the dataset to. This can be any fsspec compatible url.
        storage_options: The storage options to pass to fsspec.
    """
    if sns is None:
        raise ImportError("Please install seaborn to plot the data distribution.")

    if storage_options is None:
        storage_options = {}

    group_names = get_group_variable_names(variables)
    variable_types = get_variable_type_dict(variables)
    data.update_(
        {
            k: torch.argmax(v, dim=-1, keepdim=False)
            for k, v in data.items()
            if variable_types[k] == VariableTypeEnum.CATEGORICAL
        }
    )
    data_dict = {k: v.numpy().squeeze() for k, v in expand_tensordict_groups(data, group_names).to_dict().items()}
    if len(data_dict) > plot_num:
        print(f"Too many variables ({len(data_dict)}) to plot. Only plotting the first {plot_num}...")

    df = pd.DataFrame(data=data_dict)
    df = df.iloc[:, :plot_num]
    sns.pairplot(data=df, kind=plot_kind)
    img_path = os.path.join(datadir, "data_distribution.png")
    with fsspec.open(img_path, mode="wb", **storage_options) as f:
        plt.savefig(f)
        print(f"Saved data distribution plot to {img_path}")


def generate_save_plot_synthetic_data(
    graph_type: str,
    function_type: str,
    datadir: str,
    num_samples_train: int,
    num_samples_test: int,
    num_interventions: int,
    variable_json_path: str | None = None,
    num_nodes: int | None = None,
    graph_file: str | None = None,
    num_edges: int | None = None,
    probs: float | None = None,
    overwrite: bool = False,
    plot_kind: str = "",
    plot_num: int = 10,
    storage_options: dict[str, str] | None = None,
) -> None:
    """Generate, save and plot synthetic data.

    This will sample a SEM from the given specifications, generate data from it, and save it to disk following the
    Causica dataset format. It optionally plots the joint distribution over the variables.


    Args:
        graph_type: The type of graph to generate. Either "er" for Erdos-Renyi or "numpy" for a numpy graph.
        function_type: The type of function to generate. Either "linear" or "rff".
        datadir: The directory to save the dataset to. This can be any fsspec compatible url.
        num_samples_train: The number of training samples to generate.
        num_samples_test: The number of validation and test samples to generate.
        num_interventions: The number of interventions and counterfactuals to generate.
        variable_json_path: The path to a json file containing the variables. This can be any fsspec compatible url.
        num_nodes: The number of nodes in the graph. Not used if using a numpy graph.
        graph_file: The path to a graph file if using a numpy graph. This can be any fsspec compatible url.
        num_edges: The number of edges in the graph. Not used if using a numpy graph.
        probs: A float of the probability that an edge exists between 2 nodes
        overwrite: Whether to overwrite the dataset if it already exists.
        plot_kind: Type of joint plot to create.
        plot_num: Maximum number of variables to plot.
        storage_options: The storage options to pass to fsspec.
    """
    if storage_options is None:
        storage_options = {}

    variables = get_variable_definitions(variable_json_path, num_nodes, storage_options)
    sem_sampler = generate_sem_sampler(
        variables=variables,
        graph_file=graph_file,
        num_edges=num_edges,
        probs=probs,
        graph_type=graph_type,
        function_type=function_type,
        storage_options=storage_options,
    )
    sem = sem_sampler.sample()

    dataset = sample_dataset(
        sem=sem,
        sample_dataset_size=torch.Size([num_samples_train + num_samples_test * 2]),
        num_interventions=num_interventions,
        num_intervention_samples=num_samples_test,
        sample_interventions=True,
        sample_counterfactuals=True,
    )

    print(f"Saving dataset to {datadir}...")
    save_dataset(
        datadir,
        variables,
        sem.graph,
        dataset.observations[:num_samples_train],
        dataset.observations[num_samples_train : num_samples_train + num_samples_test],
        dataset.observations[num_samples_train + num_samples_test :],
        dataset.interventions,
        dataset.counterfactuals,
        overwrite=overwrite,
        **storage_options,
    )

    if plot_kind and plot_num > 0:
        plot_dataset(dataset.observations, variables, plot_kind, plot_num, datadir, storage_options)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable-json", "-v", type=str)
    parser.add_argument("--num-samples-train", "-n", type=int, default=1000)
    parser.add_argument("--num-samples-test", "-t", type=int, default=1000)
    parser.add_argument("--num-interventions", "-i", type=int, default=1)
    parser.add_argument("--num-variables", "-nv", type=int)
    parser.add_argument("--num-edges", "-ne", type=int)
    parser.add_argument("--edge-probabilities", "-p", type=float)
    parser.add_argument(
        "--graph-type",
        "-g",
        type=str.lower,
        default="er",
        choices=["er", "numpy"],
        help="One of er or numpy to load fixed adj matrix in npy format.",
    )
    parser.add_argument("--graph_file", "-gf", type=str)
    parser.add_argument("--seed", "-s", type=int)
    parser.add_argument("--datadir", "-d", type=str)
    parser.add_argument(
        "--function-type",
        "-f",
        type=str.lower,
        default="linear",
        choices=["linear", "rff"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot-kind", type=str, default="kde")
    parser.add_argument("--plot-num", type=int, default=10)
    parser.add_argument(
        "--storage-options",
        type=str,
        action="append",
        help="Storage options for fsspec. Options should be in the form key=value. For example --storage-options 'anon=True'",
    )

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    storage_options = None
    if args.storage_options:
        storage_options = {
            k: v.replace('"', "").replace("'", "").strip() for t in args.storage_options for k, v in [t.split("=")]
        }

        for k, v in storage_options.items():
            if v.lower() in ("true", "false"):
                storage_options[k] = v.lower() == "true"

    generate_save_plot_synthetic_data(
        variable_json_path=args.variable_json,
        num_nodes=args.num_variables,
        graph_file=args.graph_file,
        num_edges=args.num_edges,
        probs=args.edge_probabilities,
        graph_type=args.graph_type,
        datadir=args.datadir,
        function_type=args.function_type,
        num_samples_train=args.num_samples_train,
        num_samples_test=args.num_samples_test,
        num_interventions=args.num_interventions,
        overwrite=args.overwrite,
        plot_kind=args.plot_kind,
        plot_num=args.plot_num,
        storage_options=storage_options,
    )


if __name__ == "__main__":
    main()
