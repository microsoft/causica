import argparse
import os
import random
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import igraph as ig
import jax
import jax.numpy as jnp
import numpy as np

# pylint: disable=wrong-import-position
jax.config.update("jax_platform_name", "cpu")
import numpyro
import numpyro.distributions as dist

from causica.data_generation.csuite.pyro_utils import generate_dataset
from causica.data_generation.csuite.simulate import extract_observations
from causica.data_generation.csuite.utils import finalise
from causica.data_generation.large_synthetic.data_utils import sample_function, seed_iterator, simulate_dag
from causica.datasets.intervention_data import InterventionData, InterventionDataContainer, InterventionMetadata
from causica.datasets.variables import Variable, Variables


class ConcatDistribution(dist.Distribution):
    """Concatenates multiple distributions into one."""

    def __init__(self, distributions: List[dist.Distribution], validate_args: Optional[bool] = None):
        """Distribution that concatenates multiple distributions into one.

        Args:
            distributions: List of distributions to concatenate.
            validate_args: Flag whether to validate the arguments. Defaults to None.
        """
        self.distributions = distributions

        batch_shape = self.distributions[0].batch_shape

        if validate_args:
            assert all(distribution.batch_shape == batch_shape for distribution in self.distributions)
            assert all(
                np.isscalar(distribution.event_shape) or np.array(distribution.event_shape).ndim in (0, 1)
                for distribution in self.distributions
            ), f"not scalar: {[distribution.event_shape for distribution in self.distributions]}"
        event_shape = (sum(distribution.event_shape[0] for distribution in self.distributions),)

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key: jax.random.KeyArray, sample_shape: Optional[Iterable[int]] = None) -> jnp.ndarray:
        """Samples from the distribution.

        Args:
            key: Random key.
            sample_shape: Shape to use for sampling. Defaults to None.

        Returns:
            Samples from the distribution.
        """
        split_keys = jax.random.split(key, len(self.distributions))
        return jnp.concatenate([dist.sample(k, sample_shape) for k, dist in zip(split_keys, self.distributions)], -1)


def sample_distribution_parameters(variables: Iterable[Variable], continuous_noise: str = "normal") -> jnp.ndarray:
    """Samples distributions for the given variables.

    Args:
        variables: Variable definitions to sample distributions for.
        continuous_noise: Type of noise distribution to use for continuous variables ["normal" or "exponential"]. Defaults to "normal".

    Returns:
        List of distributions for the given variables.
    """
    distribution_parameter_list = []
    for variable in variables:
        if variable.type_ == "binary":
            distribution_parameter_list.append(jnp.array(np.random.uniform(size=[1, 1])))
        elif variable.type_ == "categorical":
            size = variable.upper - variable.lower + 1
            logits = np.random.normal(scale=5, size=(size,))
            distribution_parameter_list.append(jnp.array(logits.reshape([1, size])))
        elif variable.type_ == "continuous":
            if continuous_noise == "normal":
                distribution_parameter_list.append(jnp.zeros([1, 1]))
            elif continuous_noise == "exponential":
                distribution_parameter_list.append(-jnp.ones([1, 1]))
            else:
                raise ValueError(f"Continuous noise {continuous_noise} is not supported.")
        else:
            raise ValueError(f"Variable type {variable.type_} is not supported.")

    return jnp.concatenate(distribution_parameter_list, 1)


def get_distribution_from_distribution_parameters(
    variables: Iterable[Variable], distribution_parameters: jnp.ndarray, continuous_noise: str = "normal"
) -> List[dist.Distribution]:
    """Creates distributions from the given prediction of distribution parameters.

    Args:
        variables: Variables to create distributions for.
        prediction: Prediction of distribution parameters.
        continuous_noise: Type of noise distribution to use for continuous variables ["normal" or "exponential"]. Defaults to "normal".

    Returns:
        List of distributions for the given variables.
    """
    distribution_list = []
    start_idx = 0
    for variable in variables:
        end_idx = start_idx + variable.processed_dim
        cur_distribution_parameters = distribution_parameters[:, start_idx:end_idx]
        if variable.type_ == "binary":
            distribution_list.append(dist.Bernoulli(logits=cur_distribution_parameters).to_event(1))
        elif variable.type_ == "categorical":
            distribution_list.append(
                dist.Categorical(logits=cur_distribution_parameters[:, jnp.newaxis, :]).to_event(1)
            )
        elif variable.type_ == "continuous":
            if continuous_noise == "normal":
                distribution_list.append(dist.Normal(loc=cur_distribution_parameters).to_event(1))
            elif continuous_noise == "exponential":
                distribution_list.append(
                    dist.TransformedDistribution(
                        dist.Exponential(rate=jnp.ones([1, 1])).to_event(1),
                        dist.transforms.AffineTransform(loc=cur_distribution_parameters, scale=1.0),
                    )
                )
            else:
                raise ValueError(f"Continuous noise {continuous_noise} is not supported.")
        else:
            raise ValueError(f"Variable type {variable.type_} is not supported.")

        start_idx = end_idx

    return distribution_list


def sort_variables_by_group(variables: Variables) -> Variables:
    """Sorts variables by group name.

    Args:
        variables: Variables to sort.

    Returns:
        Sorted variables.
    """
    groups = defaultdict(list)
    for var in variables:
        groups[var.group_name].append(var)

    return Variables([var for group in groups.values() for var in group], None, None)


def create_model_for_variables(
    variables: Variables,
    num_edges: int,
    graph_type: str,
    graph_file: Optional[str] = None,
    sem_type: str = "linear",
    np_seed: seed_iterator = seed_iterator(),
    continuous_noise: str = "normal",
) -> Tuple[np.ndarray, Dict[str, List], Callable]:
    """Creates a numpyro model for the given variables.

    Args:
        variables: Variables to create a numpyro model for.
        num_edges: Number of edges in graph.
        graph_type: Type of graph to use ["numpy", "ER", "BP", "SF"].
        graph_file: File with graph to load if graph_type == "numpy". Defaults to None.
        sem_type: Type of functions to use for sem. Use ["linear", "mlp"]. Defaults to "linear".
        np_seed: Iterator over seeds. Defaults to seed_iterator().
        continuous_noise: Type of noise distribution to use for continuous variables ["normal" or "exponential"]. Defaults to "normal".

    Returns:
        Tuple of (adjacency matrix, dict of group names to variable indices, numpyro model).
    """
    group_names = list(dict.fromkeys([var.group_name for var in variables]))
    group_idxs = [
        [idx for idx, var in enumerate(variables) if var.group_name == group_name] for group_name in group_names
    ]
    var_to_group = {}
    num_groups = len(group_names)

    if graph_type != "numpy":
        adj_matrix = simulate_dag(np_seed, num_groups, num_edges, graph_type)
    else:
        assert isinstance(graph_file, str), "graph_file must be specified if graph_type == numpy"
        adj_matrix = np.load(graph_file)

    causal_order = ig.Graph.Adjacency(adj_matrix.T.tolist()).topological_sorting()
    # We don't convert to one-hot here
    group_shape = [len(group_idx) for group_idx in group_idxs]
    processed_group_shape = []

    parent_groups: List[List[int]] = []
    parent_variables: List[List[int]] = []

    group_functions: List[Callable[..., jnp.ndarray]] = []
    root_node_distribution_parameters: Dict[int, jnp.ndarray] = {}

    for group_idx in causal_order:
        variable_idxs = group_idxs[group_idx]
        var_to_group.update({variable_idx: group_idx for variable_idx in variable_idxs})
        cur_parent_groups = np.argwhere(adj_matrix[group_idx] > 0).flatten().tolist()
        cur_parents = [idx for group in np.array(group_idxs, dtype=object)[cur_parent_groups].tolist() for idx in group]

        parent_variables.append(cur_parents)
        parent_groups.append(cur_parent_groups)
        processed_group_shape.append(sum(variables[idx].processed_dim for idx in variable_idxs))

        if len(parent_variables[-1]) > 0:
            out_dim = processed_group_shape[-1]

            in_dim = sum(group_shape[i] for i in parent_groups[-1])

            group_functions.append(sample_function(sem_type, in_dim, np_seed, out_dim))
        else:
            root_node_distribution_parameters[group_idx] = sample_distribution_parameters(
                [variables[idx] for idx in variable_idxs], continuous_noise
            )
            group_functions.append(lambda idx: root_node_distribution_parameters[idx])

    variables_dict: Dict[str, List] = {
        "used_cols": list(range(len(variables))),
        "auxiliary_variables": [],
        "variables": [],
    }
    for i, variable in enumerate(variables):
        variables_dict["variables"].append(
            {
                "always_observed": True,
                "group_name": f"group_{var_to_group[i]}",
                "name": f"x{i}",
                "query": True,
                "type": variable.type_,
            },
        )

    def numpyro_model():
        variable_samples = {}
        for cur_idx, group_idx in enumerate(causal_order):
            variable_idxs = group_idxs[group_idx]
            cur_parent_groups = parent_groups[cur_idx]

            if len(cur_parent_groups) > 0:
                parent_variable_values = [
                    jnp.array(variable_samples[f"x{group_idx}"]) for group_idx in cur_parent_groups
                ]

                # align parent shapes in case one of them is intervened on and has no batch dimension
                num_samples = max(v.shape[0] if v.ndim == 2 else 1 for v in parent_variable_values)
                parent_variable_values = [
                    v if v.ndim == 2 else v[jnp.newaxis, :].repeat(num_samples, 0) for v in parent_variable_values
                ]
                parent_array = jnp.concatenate(parent_variable_values, -1)
            else:
                parent_array = group_idx

            distribution_parameters = group_functions[cur_idx](parent_array)
            distributions = get_distribution_from_distribution_parameters(
                [variables[idx] for idx in variable_idxs], distribution_parameters, continuous_noise
            )

            assert len(variable_idxs) == len(
                distributions
            ), f"something went wrong: variable_idxs {len(variable_idxs)} - distributions {len(distributions)}"

            group_distribution = ConcatDistribution(distributions, True)
            numpyro_sample = numpyro.sample(f"x{group_idx}", group_distribution)
            variable_samples[f"x{group_idx}"] = numpyro_sample

    return adj_matrix, variables_dict, numpyro_model


def generate_interventions(
    variables: Union[Variables, List[Variable]],
    adj_matrix: np.ndarray,
    num_interventions: int,
    num_samples: Optional[int] = None,
) -> List[Tuple[Dict, Dict, int, int]]:
    """Generates interventions for a given graph.

    Args:
        variables: Variables to generate interventions for.
        adj_matrix: Adjacency matrix of the graph.
        num_interventions: Number of interventions to generate.
        num_samples: Number of samples to generate. Defaults to None.

    Returns:
        List of interventions.
    """
    graph: ig.GraphBase = ig.Graph.Adjacency(adj_matrix.tolist())
    causal_order = graph.topological_sorting()

    group_names = list(dict.fromkeys([var.group_name for var in variables]))
    group_idxs = [
        [idx for idx, var in enumerate(variables) if var.group_name == group_name] for group_name in group_names
    ]
    group_start_idxs = [0] + list(np.cumsum([len(group) for group in group_idxs]))
    effect_idxs = [
        list(range(group_start_idxs[i], group_start_idxs[i] + len(group))) for i, group in enumerate(group_idxs)
    ]

    intervention_value_shape = [num_samples, 1] if num_samples else [1]

    interventions: List[Tuple[Dict, Dict, int, int]] = []
    while len(interventions) < num_interventions:
        intervention_dict = {}
        reference_dict = {}
        order_idx = np.random.choice(len(causal_order) // 2) + (len(causal_order) // 2)
        effect_idx = causal_order[order_idx]
        distance = 1
        while not graph.are_connected(causal_order[order_idx - distance], effect_idx) and order_idx - distance > 0:
            distance += 1

        if order_idx - distance == 0:
            continue

        group_intervention_idx = causal_order[order_idx - distance]

        variable_intervention_idxs = group_idxs[group_intervention_idx]

        intervention_list = []
        reference_list = []

        for idx in variable_intervention_idxs:
            variable = variables[idx]
            if variable.type_ == "binary":
                intervention_list.append(np.ones(intervention_value_shape))
                reference_list.append(np.zeros(intervention_value_shape))
            elif variable.type_ == "categorical":
                size = variable.upper - variable.lower + 1
                intervention_list.append(np.zeros(intervention_value_shape))
                reference_list.append(np.ones(intervention_value_shape) * (size - 1))
            elif variable.type_ == "continuous":
                intervention_list.append(np.ones(intervention_value_shape))
                reference_list.append(np.zeros(intervention_value_shape))
            else:
                raise ValueError(f"Variable type {variable.type_} is not supported.")

        intervention_dict[f"x{group_intervention_idx}"] = np.concatenate(intervention_list, -1)
        reference_dict[f"x{group_intervention_idx}"] = np.concatenate(reference_list, -1)

        interventions.append((intervention_dict, reference_dict, group_intervention_idx, effect_idx))

        print(
            f"Generated an intervention for group {group_intervention_idx} ({effect_idxs[group_intervention_idx]}) "
            + f"and effect {effect_idx} ({effect_idxs[effect_idx]}) with {list(intervention_dict.values())[0].shape}"
        )

    return interventions


def generate_synthetic_data(
    variable_json: str,
    num_edges: int,
    graph_type: str,
    graph_file: Optional[str],
    datadir: str,
    num_samples_train: int,
    num_samples_test: int,
    num_interventions: int,
    np_seed: seed_iterator = seed_iterator(),
    sem_type: str = "linear",
    continuous_noise: str = "normal",
):
    """Generates synthetic data for a given variable specification.

    Args:
        variable_json: Path to the variable specification.
        num_edges: Number of edges in graph.
        graph_type: Type of graph to use ["numpy", "ER", "BP", "SF"].
        graph_file: File with graph to load if graph_type == "numpy". Defaults to None.
        datadir: Directory to save data to.
        num_samples_train: Number of samples to generate for training.
        num_samples_test: Number of samples to generate for testing.
        num_interventions: Number of interventions to generate.
        sem_type: Type of functions to use for sem. Use ["linear", "mlp"]. Defaults to "linear".
        np_seed: Iterator over seeds. Defaults to seed_iterator().
        continuous_noise: Type of noise distribution to use for continuous variables ["normal" or "exponential"]. Defaults to "normal".
    """
    variables = Variables.create_from_json(variable_json)
    variables = sort_variables_by_group(variables)

    adj_matrix, variables_dict, numpyro_model = create_model_for_variables(
        variables, num_edges, graph_type, graph_file, sem_type, np_seed, continuous_noise
    )

    group_names = list(dict.fromkeys([var.group_name for var in variables]))
    columns_to_nodes = [
        group_idx
        for var in variables
        for group_idx, group_name in enumerate(group_names)
        if var.group_name == group_name
    ]

    interventions = generate_interventions(variables, adj_matrix, num_interventions)

    intervention_dicts = []
    for intervention in interventions:
        intervention_dicts += [intervention[0]]
        intervention_dicts += [intervention[1]]

    (
        samples_base,
        samples_test,
        samples_val,
        intervention_samples,
        _,
        _,
    ) = generate_dataset(numpyro_model, num_samples_train, num_samples_test, 0, 0, intervention_dicts)

    train_data = extract_observations(samples_base)
    val_data = extract_observations(samples_val)
    test_data = extract_observations(samples_test)
    print(
        f"Generated train data with shape {train_data.shape} for {len(variables)} Variables and {len(adj_matrix)} groups."
    )

    intervention_envs = []

    for (
        samples_int,
        samples_ref,
        (intervention_dict, reference_dict, intervention_idx, target_idxs),
    ) in zip(intervention_samples[::2], intervention_samples[1::2], interventions):
        intervention_samples = extract_observations(samples_int)
        reference_samples = extract_observations(samples_ref)

        intervention_value = np.concatenate(list(intervention_dict.values()), -1)
        reference_value = np.concatenate(list(reference_dict.values()), -1)

        intervention_envs.append(
            InterventionData(
                intervention_idxs=np.array([intervention_idx]),
                intervention_values=intervention_value,
                test_data=intervention_samples,
                intervention_reference=np.array(reference_value),
                reference_data=reference_samples,
                effect_idxs=np.array(target_idxs),
            )
        )
    intervention_data_container = InterventionDataContainer(
        InterventionMetadata(columns_to_nodes),
        intervention_envs,
    )
    intervention_data_container.validate()

    os.makedirs(datadir, exist_ok=True)

    finalise(datadir, train_data, test_data, val_data, adj_matrix, intervention_data_container, None, variables_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable_json", "-v", type=str)
    parser.add_argument("--num_samples_train", "-n", type=int, default=1000)
    parser.add_argument("--num_samples_test", "-t", type=int, default=1000)
    parser.add_argument("--num_interventions", "-i", type=int, default=1)
    parser.add_argument("--num_edges", "-e", type=int)
    parser.add_argument(
        "--graph_type",
        "-g",
        type=str,
        default="ER",
        choices=["ER", "SF", "BP", "numpy"],
        help="One of [ER, SF, BP] or numpy to load fixed adj matrix in npy format.",
    )
    parser.add_argument("--graph_file", "-f", type=str)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--datadir", "-d", type=str)
    parser.add_argument("--semtype", "-st", type=str, default="linear", choices=["linear", "mlp", "spline"])
    parser.add_argument("--continuouse_noise", "-cn", type=str, default="normal", choices=["normal", "exponential"])

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    seed_iter = seed_iterator(args.seed)

    generate_synthetic_data(
        args.variable_json,
        args.num_edges,
        args.graph_type,
        args.graph_file,
        args.datadir,
        args.num_samples_train,
        args.num_samples_test,
        args.num_interventions,
        seed_iter,
        args.semtype,
        args.continuouse_noise,
    )
