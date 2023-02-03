import os
import random
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import igraph as ig
import numpy as np
import torch
from pyro.distributions.transforms.spline import ConditionalSpline
from pyro.nn.dense_nn import DenseNN

from causica.datasets.intervention_data import InterventionData, InterventionDataContainer, InterventionMetadata
from causica.models.deci.diagonal_flows import PiecewiseRationalQuadraticTransform
from causica.utils.io_utils import save_json
from causica.utils.nri_utils import convert_temporal_to_static_adjacency_matrix


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_permutation(M: np.ndarray) -> np.ndarray:
    """
    This will randomly permute the matrix M.
    Args:
        M: the input matrix with shape [num_node, num_node].

    Returns:
        The permuted matrix
    """
    P = np.random.permutation(np.eye(M.shape[0]))
    return P.T @ M @ P


def random_acyclic_orientation(B_und: np.ndarray) -> np.ndarray:
    """
    This will randomly permute the matrix B_und followed by taking the lower triangular part.
    Args:
        B_und: The input matrix with shape [num_node, num_node].

    Returns:
        The lower triangular of the permuted matrix.
    """
    return np.tril(random_permutation(B_und), k=-1)


def generate_single_graph(num_nodes: int, graph_type: str, graph_config: dict, is_DAG: bool = True) -> np.ndarray:
    """
    This will generate a single adjacency matrix following different graph generation methods (specified by graph_type, can be "ER", "SF", "SBM").
    graph_config specifes the additional configurations for graph_type. For example, for "ER", the config dict keys can be {"p", "m", "directed", "loop"},
    refer to igraph for details. is_DAG is to ensure the generated graph is a DAG by lower-trianguler the adj, followed by a permutation.
    Note that SBM will no longer be a proper SBM if is_DAG=True
    Args:
        num_nodes: The number of nodes
        graph_type: The graph generation type. "ER", "SF" or "SBM".
        graph_config: The dict containing additional argument for graph generation.
        is_DAG: bool indicates whether the generated graph is a DAG or not.

    Returns:
        An binary ndarray with shape [num_node, num_node]
    """
    if graph_type == "ER":
        adj_graph = np.array(ig.Graph.Erdos_Renyi(n=num_nodes, **graph_config).get_adjacency().data)
        if is_DAG:
            adj_graph = random_acyclic_orientation(adj_graph)
    elif graph_type == "SF":
        if is_DAG:
            graph_config["directed"] = True
        adj_graph = np.array(ig.Graph.Barabasi(n=num_nodes, **graph_config).get_adjacency().data)
    elif graph_type == "SBM":
        adj_graph = np.array(ig.Graph.SBM(n=num_nodes, **graph_config).get_adjacency().data)
        if is_DAG:
            adj_graph = random_acyclic_orientation(adj_graph)
    else:
        raise ValueError("Unknown graph type")

    if is_DAG or graph_type == "SF":
        # SF needs to be permuted otherwise it generates either lowtri or symmetric matrix.
        adj_graph = random_permutation(adj_graph)

    return adj_graph


def generate_temporal_graph(
    num_nodes: int, graph_type: List[str], graph_config: List[dict], lag: int, random_state: Optional[int] = None
) -> np.ndarray:
    """
    This will generate a temporal graph with shape [lag+1, num_nodes, num_nodes] based on the graph_type. The graph_type[0] specifies the
    generation type for instantaneous effect and graph_type[1] specifies the lagged effect. For re-produciable results, set random_state.
    Args:
        num_nodes: The number of nodes.
        graph_type: A list containing the graph generation type. graph_type[0] for instantaneous effect and graph_type[1] for lagged effect.
        graph_config: A list of dict containing the configs for graph generation. The order should respect the graph_type.
        lag: The lag of the graph.
        random_state: The random seed used to generate the graph. None means not setting the seed.

    Returns:
        temporal_graph with shape [lag+1, num_nodes, num_nodes]
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    temporal_graph = np.full([lag + 1, num_nodes, num_nodes], np.nan)
    # Generate instantaneous effect graph
    temporal_graph[0] = generate_single_graph(num_nodes, graph_type[0], graph_config[0], is_DAG=True)
    # Generate lagged effect graph
    for i in range(1, lag + 1):
        temporal_graph[i] = generate_single_graph(num_nodes, graph_type[1], graph_config[1], is_DAG=False)

    return temporal_graph


def extract_parents(data: np.ndarray, temporal_graph: np.ndarray, node: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function will extract the parent values from data with given graph temporal_graph. It will return the lagged parents
    and instantaneous parents.
    Args:
        data: ndarray with shape [series_length, num_nodes] or [batch, series_length, num_nodes]
        temporal_graph: A binary ndarray with shape [lag+1, num_nodes, num_nodes]

    Returns:
        instant_parent: instantaneous parents with shape [parent_dim] or [batch, parents_dim]
        lagged_parent: lagged parents with shape [lagged_dim] or [batch, lagged_dim]
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # shape [1, series_length, num_nodes]

    assert data.ndim == 3, "data should be of shape [series_length, num_nodes] or [batch, series_length, num_nodes]"
    lag = temporal_graph.shape[0] - 1
    # extract instantaneous parents
    inst_parent_node = temporal_graph[0, :, node].astype(bool)  # shape [num_parents]
    inst_parent_value = data[:, -1, inst_parent_node]  # shape [batch, parent_dim]

    # Extract lagged parents
    lagged_parent_value_list = []
    for cur_lag in range(1, lag + 1):
        cur_lag_parent_node = temporal_graph[cur_lag, :, node].astype(bool)  # shape [num_parents]
        cur_lag_parent_value = data[:, -cur_lag - 1, cur_lag_parent_node]  # shape [batch, parent_dim]
        lagged_parent_value_list.append(cur_lag_parent_value)

    lagged_parent_value = np.concatenate(lagged_parent_value_list, axis=1)  # shape [batch, lagged_dim_aggregated]

    if data.shape[0] == 1:
        inst_parent_value, lagged_parent_value = inst_parent_node.squeeze(0), lagged_parent_value.squeeze(0)

    return inst_parent_value, lagged_parent_value


def simulate_history_dep_noise(lagged_parent_value: np.ndarray, noise: np.ndarray, noise_func: Callable) -> np.ndarray:
    """
    This will simulate the history-dependent noise given the lagged parent value.
    Args:
        lagged_parent_value: ndarray with shape [batch, lag_parent_dim] or [lag_parent_dim]
        noise: ndarray with shape [batch,1] or [1]
        noise_func: this specifies the function transformation for noise

    Returns:
        history-dependent noise with shape [batch, 1] or [1]
    """

    assert (
        lagged_parent_value.shape[0] == noise.shape[0]
    ), "lagged_parent_value and noise should have the same batch size"
    if lagged_parent_value.ndim == 1:
        lagged_parent_value = lagged_parent_value[np.newaxis, ...]  # shape [1, lag_parent_dim]
        noise = noise[np.newaxis, ...]  # [1, 1]

    # concat the lagged parent value and noise
    input_to_gp = np.concatenate([lagged_parent_value, noise], axis=1)  # shape [batch, lag_parent_dim+1]
    history_dependent_noise = noise_func(input_to_gp)  # shape [batch, 1]

    if lagged_parent_value.shape[0] == 1:
        history_dependent_noise = history_dependent_noise.squeeze(0)

    return history_dependent_noise


def simulate_function(lag_inst_parent_value: np.ndarray, func: Callable) -> np.ndarray:
    """
    This will simulate the function given the lagged and instantaneous parent values. The random_state_value controls
    which function is sampled from gp_func, and it controls which function is used for a particular node.
    Args:
        lag_inst_parent_value: ndarray with shape [batch, lag+inst_parent_dim] or [lag+inst_parent_dim]
        random_state_value: int controlling the function sampled from gp
        func: This specifies the functional relationships

    Returns:
        ndarray with shape [batch, 1] or [1] representing the function value for the current node.
    """

    if lag_inst_parent_value.ndim == 1:
        lag_inst_parent_value = lag_inst_parent_value[np.newaxis, ...]  # shape [1, lag+inst_parent_dim]

    function_value = func(lag_inst_parent_value)  # shape [batch, 1]

    if lag_inst_parent_value.shape[0] == 1:
        function_value = function_value.squeeze(0)

    return function_value


def simulate_single_step(
    history_data: np.ndarray,
    temporal_graph: np.ndarray,
    func_list_noise: List[Callable],
    func_list: List[Callable],
    topological_order: List[int],
    is_history_dep: bool = False,
    noise_level: float = 1,
    base_noise_type: str = "gaussian",
    intervention_dict: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    This will generate the data for a particular timestamp given the history data and temporal graph.
    Args:
        history_data: History data with shape [batch, series_length, num_node] or [series_length, num_node] containing the history observations.
        temporal_graph: The binary ndarray graph with shape [lag+1, num_nodes, num_nodes].
        func_list_noise: List of function transforms for the noise variable. Shape [num_nodes]. Each function takes [batch, dim] as input and
        output [batch,1]
        func_list: List of function for each variable, shape [num_nodes]. Each func takes [batch, dim] as input and output [batch,1]
        topological_order: the topological order from source to leaf nodes, specified by temporal graph.
        is_history_dep: bool indicate if the noise is history dependent
        base_noise_type: str, support "gaussian" and "uniform"
        intervention_dict: dict holding interventions for the current time step of form {intervention_idx: intervention_value}
    Returns:
        ndarray with shape [batch,num_node] or [num_node]
    """

    assert (
        len(func_list_noise) == len(func_list) == temporal_graph.shape[1]
    ), "function list and topological_order should have the same length as the number of nodes"
    if history_data.ndim == 2:
        history_data = history_data[np.newaxis, ...]  # shape [1, series_length, num_nodes]

    if intervention_dict is None:
        intervention_dict = {}

    batch_size = history_data.shape[0]
    # iterate through the nodes in topological order
    for node in topological_order:
        # if this node is intervened on - ignore the functions and parents
        if node in intervention_dict:
            history_data[:, -1, node] = intervention_dict[node]
        else:
            # extract the instantaneous and lagged parent values
            inst_parent_value, lagged_parent_value = extract_parents(
                history_data, temporal_graph, node
            )  # [batch, inst_parent_dim], [batch, lagged_dim_aggregated]

            # simulate the noise
            if base_noise_type == "gaussian":
                Z = np.random.randn(history_data.shape[0], 1)  # [batch, 1]
            elif base_noise_type == "uniform":
                Z = np.random.rand(history_data.shape[0], 1)
            else:
                raise NotImplementedError
            if is_history_dep:
                # Extract the noise transform
                noise_func = func_list_noise[node]

                Z = simulate_history_dep_noise(lagged_parent_value, Z, noise_func)  # [batch, 1]

            # simulate the function
            lag_inst_parent_value = np.concatenate(
                [inst_parent_value, lagged_parent_value], axis=1
            )  # [batch, lag+inst_parent_dim]
            if lag_inst_parent_value.size == 0:
                func_value = np.array(0.0)
            else:
                # Extract the function relation
                func = func_list[node]
                func_value = simulate_function(lag_inst_parent_value, func)
            history_data[:, -1, node] = (func_value + noise_level * Z).squeeze(-1)  # [batch]

    X = history_data[:, -1, :]  # [batch,num_nodes]
    if batch_size == 1:
        X = X.squeeze(0)
    return X


def format_data(data: np.ndarray):
    """
    This will transform the data format to be compatiable with the temporal data loader. It will add additional column to
    indicate the series number.
    Args:
        data: ndarray with shape [batch, series_length, num_node]

    Returns:
        transformed_data with shape [batch*series_length, num_node+1]
    """
    batch_size = data.shape[0]
    transfomred_data_list = []
    for cur_batch_idx in range(batch_size):
        cur_batch = data[cur_batch_idx, ...]  # [series_length, num_node]

        cur_series_idx = cur_batch_idx * np.ones((cur_batch.shape[0], 1))  # [series_length, 1]
        cur_transformed_batch = np.concatenate([cur_series_idx, cur_batch], axis=1)  # [series_length, num_node+1]

        transfomred_data_list.append(cur_transformed_batch)
        # transfomred_data_list.append(cur_transformed_batch)
    transformed_data = np.concatenate(transfomred_data_list, axis=0)  # [batch*series_length, num_node+1]
    return transformed_data


def select_random_history(timeseries: np.ndarray, lag: int, num_samples: int = 1) -> np.ndarray:
    """Selects random history conditioning from the input timeseries. This selects a single window per input timeseries.

    Args:
        timeseries (np.ndarray): The timeseries to select subsets from. Should have shape [batch, time, variables]
        lag (int): The window length to select.
        num_samples (int): How many histories to select. This should be <= batch.

    Returns:
        np.ndarray: Selected histories of shape [num_samples, lag, variables]
    """
    # adapted this from here: https://stackoverflow.com/questions/47982894/selecting-random-windows-from-multidimensional-numpy-array-rows
    b, t, n = timeseries.shape
    assert num_samples <= b
    idx = np.random.randint(0, t - lag + 1, b)

    s0, s1, s2 = timeseries.strides
    windows = np.lib.stride_tricks.as_strided(timeseries, shape=(b, t - lag + 1, lag, n), strides=(s0, s1, s1, s2))
    if num_samples < b:
        idx2 = np.random.randint(0, b, num_samples)
        return windows[np.arange(len(idx))[idx2], idx[idx2], :, :]
    return windows[np.arange(len(idx)), idx, :, :]


def choose_random_intervention_effect(
    ig_graph: ig.GraphBase, num_nodes: int, num_interventions: int
) -> List[Tuple[int, Tuple[int, int]]]:
    """This randomly choses an intervention - effect index pair ensuring that they are connected in the causal graph.

    Args:
        ig_graph (ig.GraphBase): DAG to check for connected-ness.
        num_nodes (int): Number of variables. This is used to find the lag from the graph.
        num_interventions (int): How many pairs to randomly choose.

    Raises:
        Exception: If it cannot find enough pairs in 100 * num_interventions tries.

    Returns:
        List[Tuple[int, int]]: List of (intervention_idx, effect_idx) tuples.
    """
    max_tries = 1000 * num_interventions

    interventions: List[Tuple[int, Tuple[int, int]]] = []
    for _ in range(max_tries):
        intervention_idx = np.random.randint(num_nodes)
        effect_idx = np.random.randint(ig_graph.vcount() - num_nodes) + num_nodes

        if ig_graph.are_connected(intervention_idx, effect_idx):
            effect_time, effect_idx = divmod(effect_idx, num_nodes)

            interventions.append((intervention_idx, (effect_idx, effect_time)))

        if len(interventions) == num_interventions:
            return interventions

    raise Exception(f"Couldn't find a connected pair in {max_tries} attempts.")


def generate_cts_temporal_data(
    path: str,
    series_length: int,
    burnin_length: int,
    num_train_samples: int,
    num_test_samples: int,
    num_nodes: int,
    graph_type: List[str],
    graph_config: List[dict],
    lag: int,
    is_history_dep: bool = False,
    noise_level: float = 1,
    function_type: str = "spline",
    noise_function_type: str = "spline",
    save_data: bool = True,
    base_noise_type: str = "gaussian",
    num_interventions: int = 0,
    num_intervention_samples: Optional[int] = None,
    intervention_value: np.ndarray = np.array(0.0),
    intervention_history: Optional[int] = None,
) -> np.ndarray:
    """
    This will generate continuous time-series data (with history-depdendent noise). It will start to collect the data after the burnin_length for stationarity.
    Args:
        path: The output dir path.
        series_length: The time series length to be generated.
        burnin_length: The burnin length before collecting the data.
        num_train_samples: The batch size of the time series data.
        num_test_samples: Number of test samples
        num_nodes: the number of variables within each timestamp.
        graph_type: The list of str specifying the instant graph and lagged graph types. graph_type[0] is for instant and
        graph_type[1] is for lagged. The choices are "ER", "SF" and "SBM".
        graph_config: A list of dict containing the graph generation configs. It should respect the order in graph_type.
        is_history_dep: bool to indicate whether the history-dependent noise is considered. Otherwise, it is Gaussian noise.
        noise_level: the std of the Gaussian noise.
        function_type: the type of function to be used for SEM. Support "spline"
        noise_function_type: the type of function to be used for history dependent noise. Support "spline_product", "conditional_spline".
        save_data: whether to save the data.
        base_noise_type: str, the base noise distribution, supports "gaussian" and "uniform"
        num_interventions: how many interventions to generate
        intervention_value: Value to use for interventions (reference = -intervention_value)
        intervention_history: Length of history length to include. Must be >= lag.
    Returns:
        None, but the stored ndarray has shape [batch*series_length, num_nodes+1], where the +1 is index of time series.
    """
    if num_intervention_samples is None:
        num_intervention_samples = num_test_samples

    num_all_samples = num_train_samples + num_test_samples

    if intervention_history is None:
        intervention_history = lag

    assert intervention_history >= lag
    # Generate graphs
    temporal_graph = generate_temporal_graph(
        num_nodes=num_nodes, graph_type=graph_type, graph_config=graph_config, lag=lag, random_state=None
    )
    # Build the function and noise_function list
    func_list, noise_func_list = build_function_list(
        temporal_graph, function_type=function_type, noise_function_type=noise_function_type
    )
    # Start data gen
    X_all = np.full((num_all_samples, burnin_length + series_length + lag, num_nodes), np.nan)
    X_all[..., 0:lag, :] = (
        np.random.randn(num_all_samples, lag, num_nodes) if num_all_samples > 1 else np.random.randn(lag, num_nodes)
    )
    # Find topological order of instant graph
    ig_graph = ig.Graph.Adjacency(temporal_graph[0].tolist())
    topological_order = ig_graph.topological_sorting()
    single_step = partial(
        simulate_single_step,
        temporal_graph=temporal_graph,
        func_list=func_list,
        func_list_noise=noise_func_list,
        topological_order=topological_order,
        is_history_dep=is_history_dep,
        noise_level=noise_level,
        base_noise_type=base_noise_type,
    )

    full_time_graph = ig.Graph.Adjacency(
        convert_temporal_to_static_adjacency_matrix(temporal_graph, "full_time").tolist()
    )

    for time in range(lag, burnin_length + series_length + lag):
        X_all[..., time, :] = single_step(history_data=X_all[..., time - lag : time + 1, :])

    # extract the stationary part of the data
    X_stationary = X_all[..., lag + burnin_length :, :]  # shape [batch, series_length, num_nodes]

    if X_stationary.ndim == 2:
        X_stationary = X_stationary[np.newaxis]

    X_train = X_stationary[:num_train_samples]
    X_test = X_stationary[num_train_samples:]

    if num_interventions > 0:
        print("simulating interventions")
        columns_to_nodes = list(range(num_nodes))
        intervention_idx_pairs = choose_random_intervention_effect(full_time_graph, num_nodes, num_interventions)

        intervention_envs = []
        intervention_value = np.array([intervention_value])
        reference_value = -intervention_value
        # generate interventions
        for intervention_idx, target_idxs in intervention_idx_pairs:
            raw_intervention_data = np.zeros((num_intervention_samples, lag * 2 + 1, num_nodes))

            history = select_random_history(X_test, intervention_history, 1)[0]

            raw_intervention_data[:, :lag] = history[:-lag]
            # generate intervention samples
            for time in range(lag + 1):
                # perform intervention only at t=0
                intervention_dict = {intervention_idx: intervention_value} if time == 0 else None
                raw_intervention_data[:, time + lag] = single_step(
                    history_data=raw_intervention_data[..., time : time + lag + 1, :],
                    intervention_dict=intervention_dict,
                )

            intervention_samples = raw_intervention_data[:, lag:].copy()

            # generate reference samples
            for time in range(lag + 1):
                # perform intervention only at t=0
                intervention_dict = {intervention_idx: reference_value} if time == 0 else None
                raw_intervention_data[:, time + lag] = single_step(
                    history_data=raw_intervention_data[..., time : time + lag + 1, :],
                    intervention_dict=intervention_dict,
                )

            reference_samples = raw_intervention_data[:, lag:].copy()

            intervention_envs.append(
                InterventionData(
                    intervention_idxs=np.array([[intervention_idx, 0]]),
                    intervention_values=intervention_value,
                    test_data=intervention_samples,
                    intervention_reference=np.array(reference_value),
                    reference_data=reference_samples,
                    effect_idxs=np.array([target_idxs]),
                    conditioning_values=history,
                    conditioning_idxs=np.array(columns_to_nodes),
                )
            )
            print(
                f"generated intervention {intervention_envs[-1].intervention_idxs}=[{intervention_value}; {reference_value}]"
            )
            int_mean = intervention_samples[:, target_idxs[1], target_idxs[0]].mean()
            ref_mean = reference_samples[:, target_idxs[1], target_idxs[0]].mean()
            print(f"effect is {target_idxs}={int_mean - ref_mean} ({int_mean} - {ref_mean})")

        intervention_data_container = InterventionDataContainer(
            InterventionMetadata(columns_to_nodes),
            intervention_envs,
        )
        intervention_data_container.validate()

    # Save the data
    if save_data:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print("[WARNING]: The save directory already exists. This might lead to unexpected behaviour.")

        np.savetxt(os.path.join(path, "train.csv"), format_data(X_train), delimiter=",")
        np.savetxt(os.path.join(path, "test.csv"), format_data(X_test), delimiter=",")
        # Save the adjacency_matrix
        np.save(os.path.join(path, "adj_matrix.npy"), temporal_graph)
        if num_interventions > 0:
            save_json(intervention_data_container.to_dict(), os.path.join(path, "interventions.json"))

    return X_stationary


def build_function_list(
    temporal_graph: np.ndarray, function_type: str, noise_function_type: str
) -> Tuple[List[Callable], List[Callable]]:
    """
    This will build two lists containing the SEM functions and history-dependent noise function, respectively.
    Args:
        temporal_graph: The input temporal graph.
        function_type: The type of SEM function used.
        noise_function_type: The tpe of history-dependent noise transformation used.

    Returns:
        function_list: list of SEM functions
        noise_function_list: list of history-dependent noise transformation
    """
    num_nodes = temporal_graph.shape[1]
    # get func_list
    function_list = []
    for cur_node in range(num_nodes):
        # get input dim
        input_dim = sum(temporal_graph[lag, :, cur_node] for lag in range(temporal_graph.shape[0])).sum().astype(int)  # type: ignore

        if input_dim == 0:
            function_list.append(zero_func)
        else:
            function_list.append(sample_function(input_dim, function_type=function_type))
    # get noise_function_list
    noise_function_list = []
    for cur_node in range(num_nodes):
        # get input dim
        input_dim = (
            sum(temporal_graph[lag, :, cur_node] for lag in range(1, temporal_graph.shape[0])).sum() + 1  # type: ignore
        ).astype(int)
        noise_function_list.append(sample_function(input_dim, function_type=noise_function_type))

    return function_list, noise_function_list


def sample_function(input_dim: int, function_type: str) -> Callable:
    """
    This will sample a function given function type.
    Args:
        input_dim: The input dimension of the function.
        function_type: The type of function to be sampled.
    Returns:
        A function sampled.
    """
    if function_type == "spline":
        return sample_spline(input_dim)
    if function_type == "spline_product":
        return sample_spline_product(input_dim)
    elif function_type == "conditional_spline":
        return sample_conditional_spline(input_dim)
    elif function_type == "mlp":
        return sample_mlp(input_dim)
    elif function_type == "inverse_noise_spline":
        return sample_inverse_noise_spline(input_dim)
    elif function_type == "mlp_noise":
        return sample_mlp_noise(input_dim)
    else:
        raise ValueError(f"Unsupported function type: {function_type}")


def sample_inverse_noise_spline(input_dim):
    flow = PiecewiseRationalQuadraticTransform(1, 16, 5, 1)
    W = np.ones((input_dim - 1, 1)) * (1 / (input_dim - 1))

    def func(X):
        z = X[..., :-1] @ W
        with torch.no_grad():
            y, _ = flow(torch.from_numpy(z))

            return -y.numpy() * X[..., -1:]

    return func


def sample_conditional_spline(input_dim):
    # input_dim is lagged_parent + 1
    noise_dim = 1
    count_bins = 8
    param_dim = [noise_dim * count_bins, noise_dim * count_bins, noise_dim * (count_bins - 1)]
    hypernet = DenseNN(input_dim - 1, [20, 20], param_dim)
    transform = ConditionalSpline(hypernet, noise_dim, count_bins=count_bins, order="quadratic")

    def func(X):
        """
        X: lagged parents concat with noise. X[...,0:-1] lagged parents, X[...,-1] noise.
        """
        with torch.no_grad():
            X = torch.from_numpy(X).float()
            transform_cond = transform.condition(X[..., :-1])
            noise_trans = transform_cond(X[..., -1:])  # [batch, 1]
        return noise_trans.numpy()

    return func


def sample_spline(input_dim):
    flow = PiecewiseRationalQuadraticTransform(1, 16, 5, 1)
    W = np.ones((input_dim, 1)) * (1 / input_dim)

    def func(X):
        z = X @ W
        with torch.no_grad():
            y, _ = flow(torch.from_numpy(z))
            return y.numpy()

    return func


def sample_mlp(input_dim):
    mlp = DenseNN(input_dim, [64, 64], [1])

    def func(X):
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            return mlp(X).numpy()

    return func


def sample_mlp_noise(input_dim):
    mlp = DenseNN(input_dim - 1, [64, 64], [1])

    def func(X):
        X_pa = torch.from_numpy(X[..., :-1]).float()
        with torch.no_grad():
            return mlp(X_pa).numpy() * X[..., -1:]

    return func


def sample_spline_product(input_dim):
    flow = sample_spline(input_dim - 1)

    def func(X):
        z_p = flow(X[..., :-1])
        out = z_p * X[..., -1:]
        return out

    return func


def zero_func() -> np.ndarray:
    return np.zeros(1)


def generate_name(
    num_nodes: int,
    graph_type: List[str],
    lag: int,
    is_history_dep: bool,
    noise_level: float,
    function_type: str,
    noise_function_type: str,
    seed: int,
    disable_inst: bool,
    connection_factor: int,
    intervention_history: int,
) -> str:
    if not is_history_dep:
        flag = "NoHistDep"
    else:
        flag = "HistDep"

    if disable_inst:
        file_name = (
            f"{graph_type[0]}_{graph_type[1]}_lag_{lag}_dim_{num_nodes}_{flag}_{noise_level}_{function_type}_"
            + f"{noise_function_type}_NoInst_con_{connection_factor}_inthist_{intervention_history}_seed_{seed}"
        )
    else:
        file_name = (
            f"{graph_type[0]}_{graph_type[1]}_lag_{lag}_dim_{num_nodes}_{flag}_{noise_level}_{function_type}_"
            + f"{noise_function_type}_con_{connection_factor}_inthist_{intervention_history}_seed_{seed}"
        )

    return file_name
