import json
import os
from enum import Enum
from typing import Any, Counter, Dict, List, Optional, Set, Tuple

import fsspec
import numpy as np
import torch
from tensordict import TensorDict

from causica.datasets.interventional_data import CounterfactualData, InterventionData

CSUITE_DATASETS_PATH = "https://azuastoragepublic.blob.core.windows.net/datasets"


InterventionsWithEffects = List[Tuple[InterventionData, InterventionData, Set[str]]]
CounterfactualsWithEffects = List[Tuple[CounterfactualData, CounterfactualData, Set[str]]]

DTYPE_MAP = {"continuous": torch.float32, "categorical": torch.int32, "binary": torch.bool}


class DataEnum(Enum):
    TRAIN = "test.csv"
    TEST = "train.csv"
    INTERVENTIONS = "interventions.json"
    COUNTERFACTUALS = "counterfactuals.json"
    TRUE_ADJACENCY = "adj_matrix.csv"
    VARIABLES_JSON = "variables.json"


def get_csuite_path(dataset_path: str, dataset_name: str, data_enum: DataEnum) -> str:
    """
    Return the the absolute path to CSuite Data from the location, dataset name and type of data.

    Args:
        dataset_path: The root path to the CSuite Data e.g. `CSUITE_DATASETS_PATH`
        dataset_name: The name of the CSuite Data e.g. "csuite_linexp_2"
        data_enum: The type of dataset for which to return the path
    Return:
        The full path to the required CSuite Dataset
    """
    return os.path.join(dataset_path, dataset_name, data_enum.value)


def load_data(
    dataset_path: str,
    dataset_name: str,
    data_enum: DataEnum,
    variables_metadata: Optional[Dict[str, Any]] = None,
    **storage_options: Dict[str, Any],
):
    """
    Load the CSuite Data from the location, dataset name and type of data.

    Args:
        dataset_path: The root path to the CSuite Data e.g. `CSUITE_DATASETS_PATH`
        dataset_name: The name of the CSuite Data e.g. "csuite_linexp_2"
        data_enum: The type of dataset for which to return the path
        variables_metadata: Optional variables object (to save downloading it multiple times)
        **storage_options: Keyword args passed to `fsspec.open`
    Return:
        The downloaded CSuite data (the type depends on the data requested)
    """

    path_name = get_csuite_path(dataset_path, dataset_name, data_enum)

    if data_enum == DataEnum.TRUE_ADJACENCY:
        with fsspec.open(path_name, **storage_options) as f:
            return np.loadtxt(f, dtype=int, delimiter=",")

    if data_enum == DataEnum.VARIABLES_JSON:
        if variables_metadata is not None:
            raise ValueError("Variables metadata was supplied and requested")
        with fsspec.open(path_name, **storage_options) as f:
            return json.load(f)

    if variables_metadata is None:
        variables_metadata = load_data(dataset_path, dataset_name, data_enum=DataEnum.VARIABLES_JSON)

    with fsspec.open(path_name, **storage_options) as f:
        if data_enum in {DataEnum.TRAIN, DataEnum.TEST}:
            arr = np.loadtxt(f, delimiter=",")
            return tensordict_from_variables_metadata(arr, variables_metadata["variables"])
        elif data_enum == DataEnum.INTERVENTIONS:
            return _load_interventions(json_object=json.load(f), metadata=variables_metadata)
        elif data_enum == DataEnum.COUNTERFACTUALS:
            with fsspec.open(path_name) as f:
                return _load_counterfactuals(json_object=json.load(f), metadata=variables_metadata)
        else:
            raise RuntimeError("Fallthrough")


def _load_interventions(json_object: Dict[str, Any], metadata: Dict[str, Any]) -> InterventionsWithEffects:
    """
    Load the CSuite Interventional Datasets as a list of interventions/counterfactuals.

    Args:
        json_object: The .json file in CSuite loaded as a json object.
        metadata: Metadata of the dataset containing names and types.

    Returns:
        A list of interventions or counterfactuals and the nodes we want to observe for each
    """
    variables_list = metadata["variables"]

    intervened_column_to_group_name: Dict[int, str] = dict(
        zip(
            json_object["metadata"]["columns_to_nodes"],
            list(item["group_name"] for item in variables_list),
        )
    )

    interventions_list = []
    for environment in json_object["environments"]:
        conditioning_idxs = environment["conditioning_idxs"]
        if conditioning_idxs is None:
            condition_nodes = []
        else:
            condition_nodes = [intervened_column_to_group_name[idx] for idx in environment["conditioning_idxs"]]

        intervention_nodes = [intervened_column_to_group_name[idx] for idx in environment["intervention_idxs"]]

        intervention = _to_intervention(
            np.array(environment["test_data"]),
            intervention_nodes=intervention_nodes,
            condition_nodes=condition_nodes,
            variables_list=variables_list,
        )
        # if the json has reference data create another intervention dataclass
        if (reference_data := environment["reference_data"]) is None:
            raise RuntimeError()
        reference = _to_intervention(
            np.array(reference_data),
            intervention_nodes=intervention_nodes,
            condition_nodes=condition_nodes,
            variables_list=variables_list,
        )

        # store the nodes we're interested in observing
        effect_nodes = set(intervened_column_to_group_name[idx] for idx in environment["effect_idxs"])
        # default to all nodes
        if not effect_nodes:
            effect_nodes = set(intervention.sampled_nodes)
        interventions_list.append((intervention, reference, effect_nodes))
    return interventions_list


def _load_counterfactuals(json_object: Dict[str, Any], metadata: Dict[str, Any]) -> CounterfactualsWithEffects:
    """
    Load the CSuite Interventional Datasets as a list of interventions/counterfactuals.

    Args:
        json_object: The .json file in CSuite loaded as a json object.
        metadata: Metadata of the dataset containing names and types.

    Returns:
        A list of interventions or counterfactuals and the nodes we want to observe for each
    """
    variables_list = metadata["variables"]

    intervened_column_to_group_name: Dict[int, str] = dict(
        zip(
            json_object["metadata"]["columns_to_nodes"],
            list(item["group_name"] for item in variables_list),
        )
    )

    cf_list = []
    for environment in json_object["environments"]:
        factual_data = np.array(environment["conditioning_values"])
        intervention_nodes = [intervened_column_to_group_name[idx] for idx in environment["intervention_idxs"]]
        intervention = _to_counterfactual(
            np.array(environment["test_data"]),
            factual_data,
            intervention_nodes=intervention_nodes,
            variables_list=variables_list,
        )
        # if the json has reference data create another intervention dataclass
        if (reference_data := environment["reference_data"]) is None:
            raise RuntimeError()
        reference = _to_counterfactual(
            np.array(reference_data), factual_data, intervention_nodes=intervention_nodes, variables_list=variables_list
        )

        # store the nodes we're interested in observing
        effect_nodes = set(intervened_column_to_group_name[idx] for idx in environment["effect_idxs"])
        # default to all nodes
        if not effect_nodes:
            effect_nodes = set(intervention.sampled_nodes)
        cf_list.append((intervention, reference, effect_nodes))
    return cf_list


def _to_intervention(
    data: np.ndarray, intervention_nodes: List[str], condition_nodes: List[str], variables_list: List[Dict[str, Any]]
) -> InterventionData:
    """Create an `InterventionData` object from the data within the json file."""
    interv_data = tensordict_from_variables_metadata(data, variables_list=variables_list)
    # all the intervention values in the dataset should be the same, so we use the first row
    first_row = interv_data[0]
    assert all(torch.allclose(interv_data[node_name], first_row[node_name]) for node_name in intervention_nodes)
    intervention_values = TensorDict(
        {node_name: first_row[node_name] for node_name in intervention_nodes}, batch_size=tuple()
    )

    condition_values = TensorDict(
        {node_name: first_row[node_name] for node_name in condition_nodes}, batch_size=tuple()
    )
    return InterventionData(
        intervention_values=intervention_values,
        intervention_data=interv_data,
        condition_values=condition_values,
    )


def _to_counterfactual(
    data: np.ndarray, base_data: np.ndarray, intervention_nodes: List[str], variables_list: List[Dict[str, Any]]
) -> CounterfactualData:
    """Create an `CounterfactualData` object from the data within the json file."""
    interv_data = tensordict_from_variables_metadata(data, variables_list=variables_list)
    # all the intervention values in the dataset should be the same, so we use the first row
    first_row = interv_data[0]
    assert all(torch.allclose(interv_data[node_name], first_row[node_name]) for node_name in intervention_nodes)
    intervention_values = TensorDict(
        {node_name: first_row[node_name] for node_name in intervention_nodes}, batch_size=tuple()
    )

    return CounterfactualData(
        intervention_values=intervention_values,
        counterfactual_data=interv_data,
        factual_data=tensordict_from_variables_metadata(base_data, variables_list=variables_list),
    )


def get_categorical_sizes(variables_list: List[Dict[str, Any]]) -> Dict[str, int]:
    categorical_sizes = {}
    for item in variables_list:
        if item["type"] == "categorical":
            upper = item.get("upper")
            lower = item.get("lower")
            if upper is not None and lower is not None:
                categorical_sizes[item["group_name"]] = item["upper"] - item["lower"] + 1
            else:
                assert upper is None and lower is None, "Please specify either both limits or neither"
                categorical_sizes[item["group_name"]] = -1
    return categorical_sizes


def tensordict_from_variables_metadata(data: np.ndarray, variables_list: List[Dict[str, Any]]) -> TensorDict:
    """
    Convert a 2D numpy array and variables information into a `TensorDict`.
    """
    assert data.ndim == 2, "Numpy loading only supported for 2d data"
    batch_size = data.shape[0]

    # guaranteed to be ordered correctly in python 3.7+ https://docs.python.org/3/library/collections.html#collections.Counter
    sizes = Counter(d["group_name"] for d in variables_list)  # get the dimensions of each key from the variables
    assert sum(sizes.values()) == data.shape[1], "Variable sizes do not match data shape"

    dtypes = {item["group_name"]: DTYPE_MAP[item["type"]] for item in variables_list}

    # slice the numpy array and assign the slices to the values of keys in the dictionary
    d = TensorDict({}, batch_size=batch_size)
    curr_idx = 0
    for key, length in sizes.items():
        d[key] = torch.Tensor(data[:, curr_idx : curr_idx + length]).to(dtype=dtypes[key])
        curr_idx += length

    return d
