import os
from typing import Dict, Optional

import graphviz
import numpy as np

from ...datasets.intervention_data import InterventionDataContainer
from ...utils.io_utils import save_json


def extract_observations(sample_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Return 2D array of extract observations from a sample dictionary."""
    samples = []
    # Iterate over variable x0, x1, ... until the next one is not found
    for idx in range(len(sample_dict)):
        name = f"x{idx}"
        if name not in sample_dict.keys():
            break
        variable_samples = sample_dict[name]
        while variable_samples.ndim < 2:
            variable_samples = np.expand_dims(variable_samples, axis=-1)
        samples.append(variable_samples)
    return np.concatenate(samples, axis=1)


def finalise(
    savedir: str,
    train_data: np.ndarray,
    test_data: np.ndarray,
    val_data: np.ndarray,
    adjacency_matrix: np.ndarray,
    intervention_container: InterventionDataContainer,
    counterfactual_container: Optional[InterventionDataContainer],
    samples_base: dict,
    override_dtypes: Optional[dict],
):

    np.savetxt(os.path.join(savedir, "adj_matrix.csv"), adjacency_matrix, delimiter=",", fmt="%i")
    np.savetxt(os.path.join(savedir, "train.csv"), train_data, delimiter=",")
    np.savetxt(os.path.join(savedir, "test.csv"), test_data, delimiter=",")
    np.savetxt(os.path.join(savedir, "val.csv"), val_data, delimiter=",")
    save_json(intervention_container.to_dict(), os.path.join(savedir, "interventions.json"))

    if counterfactual_container is not None:
        save_json(counterfactual_container.to_dict(), os.path.join(savedir, "counterfactuals.json"))

    if override_dtypes is None:
        override_dtypes = {}

    variables = []
    for name, variable_data in samples_base.items():
        for i in range(np.prod(variable_data.shape[1:], initial=1, dtype=np.int32)):
            dtype = variable_data.dtype
            if np.issubdtype(dtype, np.floating):
                inferred_type = "continuous"
            elif np.issubdtype(dtype, np.integer):
                inferred_type = "categorical"
            elif np.issubdtype(dtype, np.character):
                inferred_type = "text"
            elif np.issubdtype(dtype, bool):
                inferred_type = "binary"
            else:
                raise ValueError(f"Not recognized dtype {dtype}")
            # Group type overrides the inferred type
            group_type = override_dtypes.get(name, inferred_type)
            # Variable type overrides everything
            variable_type = override_dtypes.get(f"{name}_{i}", group_type)
            variables.append(
                {
                    "query": True,
                    "target": False,
                    "type": variable_type,
                    "name": f"{name}_{i}",
                    "group_name": name,
                    "lower": np.min(variable_data).item(),
                    "upper": np.max(variable_data).item(),
                    "always_observed": True,
                }
            )
    variables_dict = {"variables": variables, "metadata_variables": []}
    save_json(variables_dict, os.path.join(savedir, "variables.json"))
    print("Saved files to", savedir)


def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine="dot")
    names = labels if labels else [f"x{i}" for i in range(len(adjacency_matrix))]
    for name in names:
        d.node(name)
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=str(coef))
    return d


def str_to_dot(string):
    """
    Converts input string from graphviz library to valid DOT graph format.
    """
    graph = string.replace("\n", ";").replace("\t", "")
    graph = graph[:9] + graph[10:-2] + graph[-1]  # Removing unnecessary characters from string
    return graph
