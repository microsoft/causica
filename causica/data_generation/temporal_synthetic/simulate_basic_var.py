import logging
import os
from typing import Callable, Iterable

import igraph as ig
import numpy as np

NONLINEARITY = Callable[[np.ndarray], np.ndarray]
NOISEDIST = Callable[[Iterable[int]], np.ndarray]

logger = logging.getLogger(__name__)


def save_data(
    savedir: str, transition_matrix: np.ndarray, data_all: np.ndarray, train_data: np.ndarray, test_data: np.ndarray
):
    """Saves the generated data into the savedir.

    Args:
        savedir (str): Where to save the data.
        transition_matrix (np.ndarray): The (lagged) transition matrix of form [lag, variables, variables] where [:, from, to].
        data_all (np.ndarray): The full timeseries that was generated: [time, variables].
        train_data (np.ndarray): The subset of time steps used for training (these are the first in the full timeseries).
        test_data (np.ndarray): The subset of time steps used for testing (these are the last in the full timeseries).

    Raises:
        ValueError: Raises an error if the specified directory doesn't exist.
    """

    if not os.path.isdir(savedir):
        raise ValueError(f"Savedir {savedir} is not a directory!")

    np.save(os.path.join(savedir, "transition_matrix.npy"), transition_matrix)
    adjacency_matrix = np.abs(transition_matrix) > 0
    np.save(os.path.join(savedir, "adj_matrix.npy"), adjacency_matrix)

    np.savetxt(os.path.join(savedir, "all.csv"), data_all, delimiter=",")
    np.savetxt(os.path.join(savedir, "train.csv"), train_data, delimiter=",")
    np.savetxt(os.path.join(savedir, "test.csv"), test_data, delimiter=",")


def tutorial_B() -> np.ndarray:
    """Generate the same transition matrix as the VAR-LiNGaM tutorial.

    NOTE: We use the adj[from, to] convention; meaning that from causes to.

    Returns:
        np.ndarray: transition matrix.
    """

    B0 = [
        [0, -0.12, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [-0.41, 0.01, 0, -0.02, 0],
        [0.04, -0.22, 0, 0, 0],
        [0.15, 0, -0.03, 0, 0],
    ]

    B1 = [
        [-0.32, 0, 0.12, 0.32, 0],
        [0, -0.35, -0.1, -0.46, 0.4],
        [0, 0, 0.37, 0, 0.46],
        [-0.38, -0.1, -0.24, 0, -0.13],
        [0, 0, 0, 0, 0],
    ]

    return np.stack([np.array(B0).T, np.array(B1).T])


def sample_B() -> np.ndarray:
    """Generate a fixed transition matrix.

    NOTE: We use the adj[from, to] convention; meaning that from causes to.

    Returns:
        np.ndarray: transition matrix.
    """

    B0 = np.array(
        [
            [0.0, -0.12, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.41, 0.01, 0.0, -0.02, 0.0],
            [0.04, -0.22, 0.0, 0.0, 0.0],
            [0.15, 0.0, -0.03, 0.0, 0.0],
        ]
    )

    B1 = np.array(
        [
            [-0.32, 0.0, 0.12, 0.32, 0.0],
            [0.0, -0.35, -0.1, -0.46, 0.4],
            [0.0, 0.0, 0.37, 0.0, 0.46],
            [-0.38, -0.1, -0.24, 0.0, -0.13],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    B2 = np.array(
        [
            [0.0, 0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, -0.3, 0.0],
            [0.0, 0.35, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.25, 0.0],
            [0.0, 0.0, 0.0, 0.0, -0.12],
        ]
    )

    return np.stack([B0, B1, B2])


def generate_timeseries(
    B_tau: np.ndarray,
    timesteps: int,
    burnin: int = 2,
    nonlinearity: NONLINEARITY = lambda x: x,
    noise_dist: NOISEDIST = np.random.normal,  # type: ignore
) -> np.ndarray:
    """Function that generates multivariate timeseries. This assumes an additive noise model.

    Args:
        B_tau (np.ndarray): AR transition matrix of shape [lag, variables, variables].
        timesteps (int): Number of timesteps to generate.
        burnin (int, optional): Burn-in period to remove effects from random initialisation. Uses [lag * (burnin + 1)] timesteps. Defaults to 2.
        nonlinearity (NONLINEARITY, optional): Nonlinearity added after linear transition. Defaults to identity function.
        noise_dist (NOISEDIST, optional): Noise distribution used - this should probably be non-Gaussian for linear models with instantaneous effects. Defaults to np.random.normal.

    Returns:
        np.ndarray: Timeseries with [timesteps, variables] entries.
    """
    assert B_tau.ndim == 3
    assert B_tau.shape[1] == B_tau.shape[2]

    num_lag = B_tau.shape[0] - 1
    num_variables = B_tau.shape[1]

    G = ig.Graph.Weighted_Adjacency(B_tau[0])
    # Get causal ordering of variables for generation.
    causal_order = G.topological_sorting()
    print(f"Using causal order {causal_order}")
    assert len(causal_order) == num_variables

    # Pre-generate noise for additive noise model
    timeseries = noise_dist(size=(timesteps + (burnin + 1) * num_lag, num_variables))  # type: ignore

    # Iterate through timesteps and along the causal ordering to calculate each variable.
    for t in range(num_lag, timesteps + (burnin + 1) * num_lag):
        lag_start = t - num_lag

        for d in causal_order:
            timeseries[t, d] += nonlinearity(np.sum(B_tau[::-1, :, d] * timeseries[lag_start : t + 1]))

    # Crop out burn in to reduce the effect of the initial random noise in the timeseries.
    return timeseries[(burnin + 1) * num_lag :]


def generate_dataset(
    savedir: str,
    timesteps: int,
    test_timesteps: int,
    nonlinearity: NONLINEARITY = lambda x: x,
    noise_dist: NOISEDIST = np.random.normal,  # type: ignore
    B_tau_func: Callable[[], np.ndarray] = sample_B,
):
    """Generates and saves a timeseries dataset.

    Args:
        savedir (str): Save directory.
        timesteps (int): Number of training timesteps
        test_timesteps (int): Number of test timesteps
        nonlinearity (NONLINEARITY, optional): Nonlinearity added after linear transition. Defaults to identity function.
        noise_dist (NOISEDIST, optional): Noise distribution used - this should probably be non-Gaussian for linear models with instantaneous effects. Defaults to np.random.normal.
        B_tau_func (Callable[[], np.ndarray], optional): Function that generates transition matrix. Defaults to sample_B.
    """
    if os.path.exists(savedir):
        logger.warning("Generating dataset and savedir %s already exists.", savedir)

    os.makedirs(savedir, exist_ok=True)

    B_tau = B_tau_func()

    timeseries = generate_timeseries(
        B_tau, timesteps + test_timesteps, nonlinearity=nonlinearity, noise_dist=noise_dist
    )

    train_series = timeseries[:timesteps]
    test_series = timeseries[timesteps:]
    # add series index
    train_series = np.concatenate([np.zeros((train_series.shape[0], 1)), train_series], axis=1)
    test_series = np.concatenate([np.zeros((test_series.shape[0], 1)), test_series], axis=1)

    save_data(savedir, B_tau, timeseries, train_series, test_series)


if __name__ == "__main__":
    np.random.seed(42)

    generate_dataset(
        "./basic_var_tutorial",
        10000,
        500,
        noise_dist=lambda size: np.random.laplace(scale=0.5, size=size),  # type: ignore
        B_tau_func=tutorial_B,
    )

    generate_dataset(
        "./basic_var_sample",
        10000,
        500,
        noise_dist=lambda size: np.random.laplace(scale=0.5, size=size),  # type: ignore
        B_tau_func=sample_B,
    )
