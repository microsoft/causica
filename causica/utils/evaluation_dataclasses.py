from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(init=False)
class AteRMSEMetrics:
    """A class to hold average treatment effect (ATE) evaluation results.
    Args:
        group_rmses (ndarray): An array of shape (no. of interventions, no. of groups)
        containing the per-group RMSEs calculate between ground-truth and model ATE vectors.

    Attributes:
        n_interventions (int): Total number of interventions.
        n_groups (int): Total number of variable groups.
        group_rmses (ndarray): See Args.
        across_interventions (ndarray): Array of shape (no. of groups) - mean of `group_rmses` taken across interventions.
        across_groups (ndarray): Array of shape (no. of of dimensions) - mean of `group_rmses` taken across groups.
        all (np.float64): Mean of `across_interventions`.
    """

    n_interventions: int
    n_groups: int
    group_rmses: np.ndarray
    across_interventions: np.ndarray
    across_dimensions: np.ndarray
    all: np.float64

    def __init__(self, group_rmses: np.ndarray) -> None:
        self.group_rmses = group_rmses  # (no. of interventions, no. of groups)
        self.n_interventions, self.n_groups = self.group_rmses.shape
        self.across_interventions = self.group_rmses.mean(axis=0, keepdims=False)  # (no. of groups)
        self.across_groups = self.group_rmses.mean(axis=1, keepdims=False)  # (no. of interventions)
        self.all = self.across_interventions.mean(keepdims=False, dtype=np.float64)

    def get_rmse(self, intervention_idx, group_idx):
        """Returns the RMSE for the intervention `intervention_idx`
        and group `group_idx`"""
        return self.group_rmses[intervention_idx, group_idx]


@dataclass(init=False)
class IteRMSEMetrics:
    """Dataclass to hold individual treatment effect (ITE) evaluation results.
    Args:
        group_rmses (ndarray): An array of shape (no. of interventions, no. of samples, no. of groups)
            where each element corresponds to the group-wise RMSE associated with the respective
            intervention, sample and group.

    Attributes:
        group_rmses (ndarray): See Args.
        n_interventions (int): No. of interventions.
        n_samples (int): No. of samples.
        n_groups (int): No. of variable groups.
        average_ite_rmses (ndarray): Array of shape (no. of interventions, no. of groups) - mean of
            `group_rmses` taken across samples.
        across_interventions (ndarray): Array of shape (no. of groups) - mean of `average_ite_rmses` taken
            across interventions.
        across_groups (ndarray): Array of shape (no. of interventions) - mean of `average_ite_rmses` taken
            across groups.
        all (np.float64): Mean of `across_interventions`

    """

    n_interventions: int
    n_samples: int
    n_groups: int
    group_rmses: np.ndarray
    average_ite_rmses: np.ndarray
    across_interventions: np.ndarray
    across_groups: np.ndarray
    all: np.float64

    def __init__(self, group_rmses: np.ndarray) -> None:
        self.group_rmses = group_rmses  # (no. of interventions, no. of samples, no. of groups)
        self.n_interventions, self.n_samples, self.n_groups = self.group_rmses.shape
        self.average_ite_rmses = np.mean(
            self.group_rmses, axis=1, keepdims=False
        )  # (no. of interventions, no. of groups)
        self.across_interventions = np.mean(self.average_ite_rmses, axis=0, keepdims=False)  # (no. of groups)
        self.across_groups = np.mean(self.average_ite_rmses, axis=1, keepdims=False)  # (no. of interventions)
        self.all = np.mean(self.across_interventions, keepdims=False, dtype=np.float64)

    def get_rmse(self, intervention_idx, group_idx):
        """Returns the sample average RMSE for the intervention `intervention_idx`
        and group `group_idx`"""
        return self.average_ite_rmses[intervention_idx, group_idx]


@dataclass
class TreatmentDataLogProb:
    """Dataclass to hold statistics about the log-probability of test-points sampled from intervened distributions."""

    all_mean: np.ndarray
    all_std: np.ndarray
    per_intervention_mean: List[float]
    per_intervention_std: List[float]
