import argparse
import copy
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..datasets.intervention_data import InterventionData, InterventionDataContainer, InterventionMetadata
from ..utils.io_utils import read_json_as, save_json


def get_confounder_idxs(adj_matrix: np.ndarray) -> List[int]:
    """Returns list of confounders for a given adjency matrix."""
    confounder_idxs = []
    for idx in range(adj_matrix.shape[0]):
        if is_confounder(adj_matrix, idx):
            confounder_idxs.append(idx)

    return confounder_idxs


def is_confounder(adj_matrix: np.ndarray, idx: int) -> bool:
    """Checks that the given index corresponds to a confounder.

    Args:
        adj_matrix: The adjacency matrix.
        idx: Index of variables in the adjacency matrix.

    Returns:
        Whether the given index corresponds to a confounder.
    """
    return adj_matrix[:, idx].sum() == 0 and adj_matrix[idx, :].sum() > 1


def drop_variables_from_adj_matrix(adj: np.ndarray, idxs: List[int]) -> np.ndarray:
    """Drops variables specified by their indices from the adjacency matrix.

    Args:
        adj: Adjacency matrix
        idxs: List of indices of the variables to drop.

    Returns:
        A modified adjacency matrix.
    """

    new_adj = np.copy(adj)
    new_adj = np.delete(new_adj, idxs, 0)
    new_adj = np.delete(new_adj, idxs, 1)

    return new_adj


def compute_bidirected_matrix(adj_matrix: np.ndarray, confounder_idxs: List[int]) -> np.ndarray:
    """Computes the bidirected adjacency matrix.

    Args:
        adj_matrix: Adjacency matrix.
        confounder_idxs: List of confounder indices.

    Returns:
        A bidirected matrix.
    """

    # Check if each idx in confounder_idxs is really a confounder.
    for idx in confounder_idxs:
        assert is_confounder(adj_matrix, idx), f"Variable {idx} is not a confounder."

    bidirected_adj_matrix = np.zeros_like(adj_matrix)

    for confounder_idx in confounder_idxs:
        confounded_idxs = np.nonzero(adj_matrix[confounder_idx])[0]
        for i, idx1 in enumerate(confounded_idxs[:-1]):
            for idx2 in confounded_idxs[i + 1 :]:
                bidirected_adj_matrix[idx1, idx2] = bidirected_adj_matrix[idx2, idx1] = 1

    return drop_variables_from_adj_matrix(bidirected_adj_matrix, confounder_idxs)


def move_confounder_idxs_last(adj_matrix: np.ndarray, confounder_idxs: List[int]) -> np.ndarray:
    """Updates the adjacency matrix by moving confounder indices to the end.

    Args:
        adj_matrix: Adjacency matrix.
        confounder_idxs: List of confounder indices.

    Returns:
        Updated adjacency matrix.
    """
    # Check if each idx in confounder_idxs is really a confounder.
    for idx in confounder_idxs:
        assert is_confounder(adj_matrix, idx), f"Variable {idx} is not a confounder."

    # Rearrange adjacency matrix so latent confounders come last.
    non_confounder_idxs = list(set(range(adj_matrix.shape[0])) - set(confounder_idxs))
    permutation = non_confounder_idxs + confounder_idxs

    new_adj_matrix = np.empty_like(adj_matrix)
    for i, idx1 in enumerate(permutation):
        for j, idx2 in enumerate(permutation):
            new_adj_matrix[i, j] = adj_matrix[idx1, idx2]

    return new_adj_matrix


def drop_variables_from_environment(
    counterfactual_data: List[InterventionData], variable_idxs: List[int]
) -> List[Dict[str, np.ndarray]]:
    """Updates the environment data by removing confounder information.

    Args:
        counterfactual_data: Counterfactual data.
        variable_idxs: List of variable indices to drop (as latent variables)

    Returns:
        Updated environment data.
    """
    datum_dict_list = []
    for datum in counterfactual_data:
        datum_dict = datum.to_dict()

        if datum.intervention_idxs is not None:
            new_intervention_idxs = [
                np.delete(list(range(datum.test_data.shape[1])), variable_idxs).tolist().index(idx)
                for idx in datum_dict["intervention_idxs"]
            ]
            if any(x in datum_dict["intervention_idxs"] for x in variable_idxs):
                raise ValueError("Should not intervene on latent confounders")
            else:
                datum_dict["intervention_idxs"] = new_intervention_idxs

        if datum.effect_idxs is not None:
            new_effect_idxs = [
                np.delete(list(range(datum.test_data.shape[1])), variable_idxs).tolist().index(idx)
                for idx in datum_dict["effect_idxs"]
            ]
            if any(x in datum_dict["effect_idxs"] for x in variable_idxs):
                raise ValueError("Effect should not be latent confounders")
            else:
                datum_dict["effect_idxs"] = new_effect_idxs

        if datum.conditioning_idxs is not None:
            new_conditioning_idxs = list(range(datum.test_data.shape[1] - len(variable_idxs)))
            datum_dict["conditioning_idxs"] = new_conditioning_idxs

        datum_dict["test_data"] = np.delete(datum.test_data, variable_idxs, 1).tolist()

        if datum.reference_data is not None:
            datum_dict["reference_data"] = np.delete(datum.reference_data, variable_idxs, 1).tolist()
        else:
            datum_dict["reference_data"] = None

        if datum.conditioning_values is not None:
            datum_dict["conditioning_values"] = np.delete(datum.conditioning_values, variable_idxs, 1).tolist()
        else:
            datum_dict["conditioning_values"] = None

        datum_dict_list.append(datum_dict)

    return datum_dict_list


def drop_variables_from_metadata(meta_data: InterventionMetadata, variable_idxs: List[int]) -> Dict[str, List[int]]:
    """Updates the meta data by removing confounder information.

    Args:
        meta_data: Counterfactual data.
        variable_idxs: List of variable indices to drop (as latent variables)

    Returns:
        Updated meta data.
    """

    return {"columns_to_nodes": list(range(len(meta_data.columns_to_nodes) - len(variable_idxs)))}


def drop_variables_from_intervention_data(intervention_data: Dict, variable_idxs: List[int]) -> Dict[str, List[int]]:
    """Updates the intervention data by removing confounder information.

    Args:
        intervention_data: intervention data of full observations + latents
        variable_idxs: List of variable indices to drop (as latent variables)

    Returns:
        updated interventional data of observed variables only

    """
    intervention_data_observed = copy.deepcopy(intervention_data)
    intervention_data_container = InterventionDataContainer.from_dict(intervention_data)
    intervention_data_observed["environments"] = drop_variables_from_environment(
        intervention_data_container.environments, variable_idxs
    )
    intervention_data_observed["metadata"] = drop_variables_from_metadata(
        intervention_data_container.metadata, variable_idxs
    )
    return intervention_data_observed


def main(
    datadir: str,
    savedir: str,
    confounder_idxs: Optional[List[int]] = None,
    n_remove: Optional[int] = None,
    seed: int = 1,
):
    """Modify an existing dataset to a new one with latent confounders.

    Args:
        datadir: Directory containing the original dataset.
        savedir: Directory to save the new dataset to.
        confounder_idxs: List of indices of the latent confounders.
        n_remove: Number of latent confounders to remove at random. If not specified, all confounders are removed.
        seed: Random seed for removing confounders at random.
    """
    # Create directory to store the modified dataset.
    os.makedirs(savedir, exist_ok=False)

    adj_matrix = pd.read_csv(os.path.join(datadir, "adj_matrix.csv"), header=None).to_numpy()

    # Get confounder idxs.
    if confounder_idxs is None:
        confounder_idxs = get_confounder_idxs(adj_matrix)

    np.random.seed(seed)
    if n_remove is not None:
        if n_remove > len(confounder_idxs):
            raise ValueError("n_remove must be less than or equal to the number of latent confounders")
        else:
            confounder_idxs = list(np.random.choice(confounder_idxs, size=n_remove, replace=False))

    directed_matrix = drop_variables_from_adj_matrix(adj_matrix, confounder_idxs)
    bidirected_matrix = compute_bidirected_matrix(adj_matrix, confounder_idxs)
    adj_matrix = move_confounder_idxs_last(adj_matrix, confounder_idxs)
    np.savetxt(os.path.join(savedir, "directed_adjacency_matrix.csv"), directed_matrix, delimiter=",", fmt="%i")
    np.savetxt(os.path.join(savedir, "bidirected_adjacency_matrix.csv"), bidirected_matrix, delimiter=",", fmt="%i")
    np.savetxt(os.path.join(savedir, "adj_matrix.csv"), adj_matrix, delimiter=",", fmt="%i")

    if os.path.exists(os.path.join(datadir, "interventions.json")):
        intervention_data = read_json_as(os.path.join(datadir, "interventions.json"), dict)
        intervention_data_observed = drop_variables_from_intervention_data(intervention_data, confounder_idxs)
        save_json(intervention_data_observed, os.path.join(savedir, "interventions.json"))

    if os.path.exists(os.path.join(datadir, "counterfactuals.json")):
        counterfactual_data = read_json_as(os.path.join(datadir, "counterfactuals.json"), dict)
        counterfactual_data_observed = drop_variables_from_intervention_data(counterfactual_data, confounder_idxs)
        save_json(counterfactual_data_observed, os.path.join(savedir, "counterfactuals.json"))

    for name in ["all", "train", "test"]:
        if os.path.exists(os.path.join(datadir, f"{name}.csv")):
            data = pd.read_csv(os.path.join(datadir, f"{name}.csv"), header=None).to_numpy()
            data = np.delete(data, confounder_idxs, 1)
            np.savetxt(os.path.join(savedir, f"{name}.csv"), data, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", help="Directory containing the original dataset.", type=str)
    parser.add_argument("--savedir", help="Directory to save the new dataset to.", type=str)
    parser.add_argument(
        "--idxs", help="List of indices of latent confounders to remove", nargs="+", type=int, default=None
    )
    parser.add_argument("--n_remove", help="Number of confounders to remove at random", type=int, default=None)
    parser.add_argument("--seed", help="Random seed", type=int, default=1)
    parsed_args = parser.parse_args()

    main(parsed_args.datadir, parsed_args.savedir, parsed_args.idxs, parsed_args.n_remove, parsed_args.seed)
