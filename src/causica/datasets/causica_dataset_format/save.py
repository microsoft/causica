import dataclasses
import json
import os
from typing import Any, Callable, Iterable, Optional

import fsspec
import numpy as np
import torch
from tensordict import TensorDict

from causica.datasets.causica_dataset_format.load import (
    CounterfactualWithEffects,
    DataEnum,
    InterventionWithEffects,
    Variable,
    VariablesMetadata,
    get_group_names,
)
from causica.datasets.interventional_data import CounterfactualData, InterventionData
from causica.datasets.variable_types import VariableTypeEnum


class TensorEncoder(json.JSONEncoder):
    """A JSON encoder that can encode tensors, numpy arrays and TensorDicts."""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return self.default(o.numpy())
        if isinstance(o, TensorDict):
            return o.to_dict()
        if isinstance(o, set):
            return list(o)
        return json.JSONEncoder.default(self, o)


class VariableEncoder(json.JSONEncoder):
    """A JSON encoder that can encode Variable objects."""

    def default(self, o):
        if isinstance(o, VariablesMetadata):
            return dataclasses.asdict(o)
        if isinstance(o, Variable):
            return dataclasses.asdict(o)
        if isinstance(o, VariableTypeEnum):
            return o.value
        return json.JSONEncoder.default(self, o)


def _get_idx_list_from_node_names(names: Iterable[str], node_names: Iterable[str]) -> list[int]:
    """Returns a list of indices corresponding to the node names in the node_names list.

    Args:
        names: A list of node names
        node_names: A list of all node names

    Returns:
        A list of indices corresponding to the node names in the node_names list
    """
    return [np.argwhere(np.array(node_names) == name).item() for name in names]


def _packed_concatenate_tensordict_values(
    td: TensorDict, node_names: Iterable[str], variable_types: dict[str, VariableTypeEnum]
) -> np.ndarray:
    """Concatenates the values of a TensorDict along the last dimension. The order of the values is determined by the node_names list.

    Args:
        td: A TensorDict
        node_names: The names of the nodes in the TensorDict

    Returns:
        A numpy array of the concatenated values
    """

    def _maybe_pack_onehot(t: torch.Tensor, variable_type: VariableTypeEnum):
        return torch.argmax(t, dim=-1, keepdim=True) if variable_type == VariableTypeEnum.CATEGORICAL else t

    return np.concatenate(
        [_maybe_pack_onehot(td[name], variable_types[name]) for name in node_names if name in td.keys()], axis=-1
    )


def intervention_data_to_dict(
    intervention: InterventionData,
    reference: InterventionData | None,
    variables_metadata: VariablesMetadata,
    effect_idx_names: Iterable[str] | None = None,
) -> dict[str, np.ndarray | None]:
    """Converts an InterventionData object to a dictionary that can be saved to a json file.

    Args:
        intervention: An InterventionData object
        reference: An optional InterventionData object to use as a reference
        variables_metadata: The variables metadata for the dataset
        effect_idx_names: The names of the variables to use as the effect

    Returns:
        A dictionary of a single intervention data sample in causica dataset format that can be saved to a json file
    """
    node_names = get_group_names(variables_metadata)
    variable_types = {var.group_name: var.type for var in variables_metadata.variables}

    effect_idxs = _get_idx_list_from_node_names(effect_idx_names or [], node_names)
    intervention_idxs = _get_idx_list_from_node_names(intervention.intervention_values.keys(), node_names)

    intervention_values = _packed_concatenate_tensordict_values(
        intervention.intervention_values, node_names, variable_types
    )
    test_data = _packed_concatenate_tensordict_values(intervention.intervention_data, node_names, variable_types)

    reference_data = None
    intervention_reference = None
    if reference is not None:
        reference_data = _packed_concatenate_tensordict_values(reference.intervention_data, node_names, variable_types)
        intervention_reference = _packed_concatenate_tensordict_values(
            reference.intervention_values, node_names, variable_types
        )

    return {
        "intervention_idxs": np.array(intervention_idxs),
        "intervention_values": intervention_values,
        "test_data": test_data,
        "reference_data": reference_data,
        "intervention_reference": intervention_reference,
        "effect_idxs": np.array(effect_idxs),
        "conditioning_idxs": None,
        "conditioning_values": None,
    }


def counterfactual_data_to_dict(
    counterfactual: CounterfactualData,
    reference: Optional[CounterfactualData],
    variables_metadata: VariablesMetadata,
    effect_idx_names: Iterable[str] | None = None,
) -> dict[str, np.ndarray | list[int] | None]:
    """Converts a CounterfactualData object to a dictionary that can be saved to a json file.

    Args:
        counterfactual: A CounterfactualData object
        reference: An optional CounterfactualData object to use as a reference
        variables_metadata: The variables metadata for the dataset
        effect_idx_names: The names of the variables to use as the effect

    Returns:
        A dictionary of a single counterfactual data sample in causica dataset format that can be saved to a json file
    """
    node_names = get_group_names(variables_metadata)
    variable_types = {var.group_name: var.type for var in variables_metadata.variables}

    effect_idxs = _get_idx_list_from_node_names(effect_idx_names or [], node_names)
    conditioning_idxs = _get_idx_list_from_node_names(counterfactual.factual_data.keys(), node_names)
    intervention_idxs = _get_idx_list_from_node_names(counterfactual.intervention_values.keys(), node_names)

    conditioning_values = _packed_concatenate_tensordict_values(counterfactual.factual_data, node_names, variable_types)
    intervention_values = _packed_concatenate_tensordict_values(
        counterfactual.intervention_values, node_names, variable_types
    )
    test_data = _packed_concatenate_tensordict_values(counterfactual.counterfactual_data, node_names, variable_types)

    reference_data = None
    intervention_reference = None
    if reference is not None:
        reference_data = _packed_concatenate_tensordict_values(
            reference.counterfactual_data, node_names, variable_types
        )
        intervention_reference = _packed_concatenate_tensordict_values(
            reference.intervention_values, node_names, variable_types
        )

    return {
        "intervention_idxs": intervention_idxs,
        "intervention_values": intervention_values,
        "test_data": test_data,
        "effect_idxs": effect_idxs,
        "conditioning_idxs": conditioning_idxs,
        "conditioning_values": conditioning_values,
        "reference_data": reference_data,
        "intervention_reference": intervention_reference,
    }


def intervention_or_counterfactual_to_causica_dict(
    data_with_effects: list[CounterfactualWithEffects] | list[InterventionWithEffects],
    variables_metadata: VariablesMetadata,
) -> dict[str, Any]:
    """Converts a list of CounterfactualWithEffects objects to a dictionary that can be saved to a json file.

    Args:
        data_with_effects: A list of CounterfactualWithEffects or InterventionWithEffects objects
        variables_metadata: The variables metadata for the dataset

    Returns:
        A dictionary of the data in causica dataset format that can be saved to a json file
    """
    node_names = get_group_names(variables_metadata)

    convert_fn: Callable
    if isinstance(data_with_effects[0][0], InterventionData):
        assert all(
            isinstance(data, InterventionData) and (reference is None or isinstance(reference, InterventionData))
            for data, reference, _ in data_with_effects
        )
        convert_fn = intervention_data_to_dict
        data_td = data_with_effects[0][0].intervention_data
    elif isinstance(data_with_effects[0][0], CounterfactualData):
        assert all(
            isinstance(data, CounterfactualData) and (reference is None or isinstance(reference, CounterfactualData))
            for data, reference, _ in data_with_effects
        )
        convert_fn = counterfactual_data_to_dict
        data_td = data_with_effects[0][0].factual_data
    else:
        raise ValueError(
            f"data_with_effects must be a list of InterventionData or CounterfactualData objects, got {type(data_with_effects[0][0])}"
        )

    columns_to_nodes = [idx for idx, name in enumerate(node_names) for _ in range(data_td[name].shape[-1])]

    return {
        "metadata": {
            "columns_to_nodes": columns_to_nodes,
        },
        "environments": [
            convert_fn(data_obj, reference_obj, variables_metadata, effect_idxs)
            for data_obj, reference_obj, effect_idxs in data_with_effects
        ],
    }


def save_data(
    savedir: str,
    data: VariablesMetadata
    | torch.Tensor
    | TensorDict
    | list[InterventionWithEffects]
    | list[CounterfactualWithEffects],
    data_enum: DataEnum | None = None,
    variables: VariablesMetadata | None = None,
    **storage_kwargs,
) -> None:
    """
    Save data to disk in the Causica dataset format.

    Args:
        savedir (str): The directory to save the data to. This can be any fsspec compatible url.
        data:
            The data to save. Must be one of the following types:
            - VariablesMetadata: If saving variables metadata.
            - torch.Tensor: If saving the true adjacency matrix.
            - TensorDict: If saving train, validation, or test data.
            - list[InterventionWithEffects]: If saving interventions data.
            - list[CounterfactualWithEffects]: If saving counterfactuals data.
        data_enum: The type of data being saved. Defaults to None. Not required for variables or adjacency.
        variables: The variables metadata associated with the data. Defaults to None. Not required for variables or adjacency.
        **storage_kwargs: Additional keyword arguments to pass to the fsspec.open function.

    Raises:
        AssertionError: If the input data is of an unexpected type or if the data_enum and variables arguments are not consistent with the input data type.
    """
    if isinstance(data, VariablesMetadata):
        assert data_enum is None or data_enum == DataEnum.VARIABLES_JSON
        with fsspec.open(os.path.join(savedir, "variables.json"), mode="w", **storage_kwargs) as f:
            json.dump(dataclasses.asdict(data), f, indent=4, cls=VariableEncoder)
    elif isinstance(data, torch.Tensor):
        assert data_enum is None or data_enum == DataEnum.TRUE_ADJACENCY
        with fsspec.open(os.path.join(savedir, "adj_matrix.csv"), mode="w", **storage_kwargs) as f:
            np.savetxt(
                f,
                data.numpy(),
                delimiter=",",
                fmt="%i",
            )
    elif isinstance(data, TensorDict):
        assert data_enum in (DataEnum.TRAIN, DataEnum.VALIDATION, DataEnum.TEST)
        assert variables is not None
        node_names = get_group_names(variables)
        variable_types = {var.group_name: var.type for var in variables.variables}
        with fsspec.open(os.path.join(savedir, data_enum.value), mode="w", **storage_kwargs) as f:
            np.savetxt(
                f,
                _packed_concatenate_tensordict_values(data, node_names, variable_types),
                delimiter=",",
            )
    else:
        assert (
            data_enum == DataEnum.COUNTERFACTUALS
            and isinstance(data, list)
            and isinstance(data[0][0], CounterfactualData)
        ) or (
            data_enum == DataEnum.INTERVENTIONS and isinstance(data, list) and isinstance(data[0][0], InterventionData)
        )
        assert variables is not None
        with fsspec.open(os.path.join(savedir, data_enum.value), mode="w", **storage_kwargs) as f:
            json.dump(
                intervention_or_counterfactual_to_causica_dict(data, variables),
                f,
                indent=4,
                cls=TensorEncoder,
            )


def save_dataset(
    savedir: str,
    variables: VariablesMetadata,
    adjacency: torch.Tensor,
    train_data: TensorDict,
    test_data: TensorDict,
    val_data: TensorDict | None = None,
    interventions: list[InterventionWithEffects] | None = None,
    counterfactuals: list[CounterfactualWithEffects] | None = None,
    overwrite: bool = False,
    **storage_kwargs,
) -> None:
    """
    Saves a dataset to a directory in the Causica format.

    Args:
        savedir: The directory to save the dataset to. This can be any fsspec compatible url.
        variables: Metadata about the variables in the dataset.
        adjacency: The adjacency matrix of the causal graph.
        train_data: The training data.
        test_data: The test data.
        val_data: The validation data.
        interventions: A list of interventions to save.
        counterfactuals: A list of counterfactuals to save.
        overwrite: Whether to overwrite the directory if it already exists. Defaults to False.
        **storage_kwargs: Additional keyword arguments to pass to the storage backend.
    """
    fs, _ = fsspec.core.url_to_fs(savedir, **storage_kwargs)
    if fs.exists(savedir):
        print(f"Directory {savedir} already exists...")
        if not overwrite:
            print("Skipping...")
            return
    fs.makedirs(savedir, exist_ok=True)

    save_data(savedir, variables, DataEnum.VARIABLES_JSON, variables, **storage_kwargs)
    save_data(savedir, adjacency, DataEnum.TRUE_ADJACENCY, variables, **storage_kwargs)
    save_data(savedir, train_data, DataEnum.TRAIN, variables, **storage_kwargs)
    save_data(savedir, test_data, DataEnum.TEST, variables, **storage_kwargs)
    if val_data is not None:
        save_data(savedir, val_data, DataEnum.VALIDATION, variables, **storage_kwargs)
    if interventions:
        save_data(savedir, interventions, DataEnum.INTERVENTIONS, variables, **storage_kwargs)
    if counterfactuals:
        save_data(savedir, counterfactuals, DataEnum.COUNTERFACTUALS, variables, **storage_kwargs)
