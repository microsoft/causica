import os
from typing import List, Union

import graphviz
import numpy as np

from ...utils.helper_functions import convert_dict_of_ndarray_to_lists
from ...utils.io_utils import save_json


def make_coding_tensors(
    num_samples,
    num_variables,
    do_idx,
    do_value,
    reference_value=None,
    condition_idx=None,
    condition_value=None,
    target_idxs=None,
):

    intervention = np.full((num_samples, num_variables), np.nan)
    intervention[:, do_idx] = do_value
    reference = np.full((num_samples, num_variables), np.nan)

    if reference_value is not None:
        reference[:, do_idx] = reference_value

    conditioning = np.full((num_samples, num_variables), np.nan)
    if condition_idx is not None and condition_value is not None:
        conditioning[:, condition_idx] = condition_value

    targets = np.full((num_samples, num_variables), np.nan)
    if target_idxs is not None:
        for t in target_idxs:
            targets[:, t] = 1.0

    return conditioning, intervention, reference, targets


def to_counterfactual_dict_format(
    original_samples: np.ndarray,
    intervention_samples: np.ndarray,
    reference_samples: np.ndarray,
    do_idx: List[int],
    do_value: Union[int, float],
    reference_value: Union[int, float],
    target_idxs: List[int] = None,
):
    """
    Converts data to the following format

        [{
            "conditioning": np.array[mixed],
            "intervention": np.array[mixed],
            "reference": Optional[np.array[mixed]],
            "effect_mask": Optional[np.array[bool]],
            "intervention_samples": np.array[mixed],
            "reference_samples": Optional[np.array[mixed]],
        }],

    Args:
        conditioning: An ndarray of shape (no. of samples, no. of nodes) which contains nans for all but conditioned variables
        intervention: An ndarray of shape (1, no. of nodes) which contains nans for all but intervened variables
        reference: An ndarray of shape (1, no. of nodes) which contains nans for all but intervened variables
        effect_mask: A binary ndarray of shape (1, no. of nodes) where 1's indicate an effect variable
        intervention_samples: An ndarray of shape (no. of samples, no. of nodes) containing samples produced using the 'intervention' intervention
        reference_samples: An ndarray of shape (no. of samples, no. of nodes) containing samples produced using the 'reference' intervention
    Returns: dict in counterfactual data format
    """
    assert intervention_samples.shape == reference_samples.shape == original_samples.shape

    num_variables = original_samples.shape[1]

    intervention = np.array([np.nan for _ in range(num_variables)])
    intervention[do_idx] = do_value

    reference = np.array([np.nan for _ in range(num_variables)])
    reference[do_idx] = reference_value

    effect_mask = np.array([False for _ in range(num_variables)])
    if target_idxs is not None:
        for idx in target_idxs:
            effect_mask[idx] = True

    return [
        {
            "conditioning": original_samples,
            "intervention": intervention,
            "reference": reference,
            "intervention_samples": intervention_samples,
            "reference_samples": reference_samples,
            "effect_mask": effect_mask,
        }
    ]


def extract_observations(sample_dict):
    """
    Extract observations from a sample dictionary into a 2d np array
    """
    done = False
    idx = 0
    samples = []
    while not done:
        if f"x{idx}" in sample_dict.keys():
            samples.append(sample_dict[f"x{idx}"])
            idx += 1
        else:
            done = True
    return np.stack(samples, axis=1)


def finalise(savedir, train_data, adjacency_matrix, interventions, metadata, counterfactual_dict):
    test_data = np.concatenate(interventions, axis=0)

    # Create metadata
    metadata = [np.concatenate(data, axis=0) for data in zip(*metadata)]
    metadata.append(test_data)
    metadata_matrix = np.concatenate(metadata, axis=1)

    data_all = np.concatenate([train_data, test_data], axis=0)

    np.savetxt(os.path.join(savedir, "adj_matrix.csv"), adjacency_matrix, delimiter=",", fmt="%i")
    np.savetxt(os.path.join(savedir, "all.csv"), data_all, delimiter=",")
    np.savetxt(os.path.join(savedir, "train.csv"), train_data, delimiter=",")
    np.savetxt(os.path.join(savedir, "test.csv"), test_data, delimiter=",")
    np.savetxt(os.path.join(savedir, "interventions.csv"), metadata_matrix, delimiter=",")

    if counterfactual_dict is not None:
        save_json(
            [convert_dict_of_ndarray_to_lists(c) for c in counterfactual_dict],
            os.path.join(savedir, "counterfactuals.json"),
        )


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
