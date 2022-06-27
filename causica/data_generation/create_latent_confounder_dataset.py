import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

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


def move_confounder_idxs_last(
    adj_matrix: np.ndarray, confounder_idxs: List[int]
) -> np.ndarray:
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


def drop_variables_from_interventions(
    intervention_data: np.ndarray, variable_idxs: List[int]
) -> np.ndarray:
    """Updates the intervention data by removing latent confounder information.

    Args:
        intervention_data: Intervention data.
        variable_idxs: List of variable indices.

    Returns:
        Updated intervention data.
    """
    nvars = int(intervention_data.shape[1] / 5)
    _idxs = [i * nvars + idx for idx in variable_idxs for i in range(5)]
    intervention_data = np.delete(intervention_data, _idxs, 1)

    return intervention_data


def drop_variables_from_counterfactuals(
    counterfactual_data: List[Dict[str, np.ndarray]], variable_idxs: List[int]
) -> List[Dict[str, np.ndarray]]:
    """Updates the counterfactual data by removing confounder information.

    Args:
        counterfactual_data: Counterfactual data.
        variable_idxs: List of variable indices.

    Returns:
        Update counterfactual data.
    """
    for datum in counterfactual_data:
        datum["conditioning"] = np.delete(
            datum["conditioning"], variable_idxs, 1
        ).tolist()
        datum["effect_mask"] = np.delete(
            datum["effect_mask"], variable_idxs, 0
        ).tolist()
        datum["intervention"] = np.delete(
            datum["intervention"], variable_idxs, 0
        ).tolist()
        datum["intervention_samples"] = np.delete(
            datum["intervention_samples"], variable_idxs, 1
        ).tolist()
        datum["reference"] = np.delete(datum["reference"], variable_idxs, 0).tolist()
        datum["reference_samples"] = np.delete(
            datum["reference_samples"], variable_idxs, 1
        ).tolist()

    return counterfactual_data


def main(datadir: str, savedir: str):
    """Modify an existing dataset to a new one with latent confounders.

    Args:
        datadir: Directory containing the original dataset.
        savedir: Directory to save the new dataset to.
    """
    # Create directory to store the modified dataset.
    os.makedirs(savedir, exist_ok=False)

    adj_matrix = pd.read_csv(
        os.path.join(datadir, "adj_matrix.csv"), header=None
    ).to_numpy()
    confounder_idxs = get_confounder_idxs(adj_matrix)
    adj_matrix = move_confounder_idxs_last(adj_matrix, confounder_idxs)
    np.savetxt(
        os.path.join(savedir, "adj_matrix.csv"), adj_matrix, delimiter=",", fmt="%i"
    )

    if os.path.exists(os.path.join(datadir, "intervention.csv")):
        intervention_data = pd.read_csv(
            os.path.join(datadir, "intervention.csv"), header=None
        ).to_numpy()
        intervention_data = drop_variables_from_interventions(
            intervention_data, confounder_idxs
        )
        np.savetxt(
            os.path.join(savedir, "interventions.csv"), intervention_data, delimiter=","
        )

    if os.path.exists(os.path.join(datadir, "counterfactuals.json")):
        counterfactual_data = read_json_as(
            os.path.join(datadir, "counterfactuals.json"), list
        )
        counterfactual_data = drop_variables_from_counterfactuals(
            counterfactual_data, confounder_idxs
        )
        save_json(counterfactual_data, os.path.join(savedir, "counterfactuals.json"))

    for name in ["all", "train", "test"]:
        if os.path.exists(os.path.join(datadir, f"{name}.csv")):
            data = pd.read_csv(
                os.path.join(datadir, f"{name}.csv"), header=None
            ).to_numpy()
            data = np.delete(data, confounder_idxs, 1)
            np.savetxt(os.path.join(savedir, f"{name}.csv"), data, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-datadir", help="Directory containing the original dataset.", type=str
    )
    parser.add_argument(
        "-savedir", help="Directory to save the new dataset to.", type=str
    )
    parsed_args = parser.parse_args()

    main(parsed_args.datadir, parsed_args.savedir)
