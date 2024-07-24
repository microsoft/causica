import argparse
import os
import random
from enum import Enum
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray


class CsuiteEnum(Enum):
    """Different data generation processes.
    These datasets are not meant to be used outside this project.
    """

    LINGAUSS = "lingauss"
    LINEXP = "linexp"
    NONLINGAUSS = "nonlingauss"
    NONLIN_SIMPSON = "nonlin_simpson"
    SYMPROD_SIMPSON = "symprod_simpson"
    LARGE_BACKDOOR = "large_backdoor"
    WEAK_ARROW = "weak_arrow"


def generate_observational_data(noise: NDArray, dic_func: dict[str, Callable]) -> NDArray:
    x = np.zeros(noise.shape)
    for k in range(len(dic_func)):
        x[:, k] = dic_func[f"f{k}"](x, noise)
    return x


def compute_counterfactual(noise: NDArray, dic_func: dict[str, Callable]) -> NDArray:
    # generate factual data
    x_f = generate_observational_data(noise, dic_func)

    # choose a random index
    idx_int = random.randint(0, len(dic_func) - 1)

    # choose a random value for the intervention
    min_val = min(x_f[:, idx_int])
    max_val = max(x_f[:, idx_int])
    val_int = np.random.uniform(min_val, max_val)

    # create the interventional function
    dic_func_cf = dic_func.copy()
    dic_func_cf[f"f{idx_int}"] = lambda x, z: z[:, idx_int]

    # create the counterfactual noise
    noise_cf = noise.copy()
    noise_cf[:, idx_int] = val_int

    # compute the counterfactual
    x_cf = generate_observational_data(noise_cf, dic_func_cf)

    # concatenate all the data
    x_all = np.concatenate((x_f, x_cf), axis=1)
    x_all = np.concatenate((x_all, np.array([idx_int] * len(x_all)).reshape(-1, 1)), axis=1)
    x_all = np.concatenate((x_all, np.array([val_int] * len(x_all)).reshape(-1, 1)), axis=1)

    return x_all


# Function that generates the data
def generate_lingauss(
    nsamples: int, seed: int = 49, num_interventions: int = 2, num_samples_interventions: int = 100
) -> tuple[NDArray, NDArray, NDArray, list[NDArray] | None]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Generate graph
    adjacency_matrix = np.zeros((2, 2))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix = adjacency_matrix.T

    # Generate dic of functions
    dic_func = {}
    dic_func["f0"] = lambda x, z: z[:, 0]
    dic_func["f1"] = lambda x, z: 0.5 * x[:, 0] + (np.sqrt(3) / 2) * z[:, 1]

    # Generate noise
    z = np.random.normal(loc=0.0, scale=1.0, size=(nsamples, 2))

    # Generate observational data
    x = generate_observational_data(z, dic_func)

    # Generate counterfactuals
    if num_interventions > 0:
        list_x_cf = []
        for _ in range(num_interventions):
            # Generate a new noise for each intervention
            z_f = np.random.normal(loc=0.0, scale=1.0, size=(num_samples_interventions, 2))
            x_cf = compute_counterfactual(z_f, dic_func)
            list_x_cf.append(x_cf)
    else:
        list_x_cf = None

    return x, z, adjacency_matrix, list_x_cf


def generate_linexp(
    nsamples: int, seed: int = 49, num_interventions: int = 2, num_samples_interventions: int = 100
) -> tuple[NDArray, NDArray, NDArray, list[NDArray] | None]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Generate graph
    adjacency_matrix = np.zeros((2, 2))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix = adjacency_matrix.T

    # Generate dic of functions
    dic_func = {}
    dic_func["f0"] = lambda x, z: z[:, 0]
    dic_func["f1"] = lambda x, z: 0.5 * x[:, 0] + (np.sqrt(3) / 2) * z[:, 1]

    # Generate noise
    z = np.random.exponential(scale=1.0, size=(nsamples, 2)) - 1

    # Generate observational data
    x = generate_observational_data(z, dic_func)

    # Generate counterfactuals
    if num_interventions > 0:
        list_x_cf = []
        for _ in range(num_interventions):
            # Generate a new noise for each internvention
            z_f = np.random.exponential(scale=1.0, size=(num_samples_interventions, 2)) - 1
            x_cf = compute_counterfactual(z_f, dic_func)
            list_x_cf.append(x_cf)
    else:
        list_x_cf = None

    return x, z, adjacency_matrix, list_x_cf


def generate_nonlingauss(
    nsamples: int, seed: int = 49, num_interventions: int = 2, num_samples_interventions: int = 100
) -> tuple[NDArray, NDArray, NDArray, list[NDArray] | None]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Generate graph
    adjacency_matrix = np.zeros((2, 2))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix = adjacency_matrix.T

    # Generate dic of functions
    dic_func = {}
    dic_func["f0"] = lambda x, z: z[:, 0]
    alpha = np.sqrt(1 - (6 * ((1 / np.sqrt(5)) - (1 / 3))))
    dic_func["f1"] = lambda x, z: np.sqrt(6) * np.exp(-(x[:, 0] ** 2)) + alpha * z[:, 1]

    # Generate noise
    z = np.random.normal(loc=0.0, scale=1.0, size=(nsamples, 2))

    # Generate observational data
    x = generate_observational_data(z, dic_func)

    # Generate counterfactuals
    if num_interventions > 0:
        list_x_cf = []
        for _ in range(num_interventions):
            # Generate a new noise for each internvention
            z_f = np.random.normal(loc=0.0, scale=1.0, size=(num_samples_interventions, 2))
            x_cf = compute_counterfactual(z_f, dic_func)
            list_x_cf.append(x_cf)
    else:
        list_x_cf = None

    return x, z, adjacency_matrix, list_x_cf


def generate_nonlin_simpson(
    nsamples: int, seed: int = 49, num_interventions: int = 4, num_samples_interventions: int = 100
) -> tuple[NDArray, NDArray, NDArray, list[NDArray] | None]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Generate graph
    adjacency_matrix = np.zeros((4, 4))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[0, 2] = 1
    adjacency_matrix[1, 2] = 1
    adjacency_matrix[2, 3] = 1
    adjacency_matrix = adjacency_matrix.T

    # Generate dic of functions
    dic_func = {}
    dic_func["f0"] = lambda x, z: z[:, 0]
    dic_func["f1"] = lambda x, z: np.log(1 + np.exp(1 - x[:, 0])) + np.sqrt(3 / 20) * z[:, 1]
    dic_func["f2"] = lambda x, z: np.tanh(2 * x[:, 1]) + (3 / 2) * x[:, 0] - 1 + np.tanh(z[:, 2])
    dic_func["f3"] = lambda x, z: 5 * np.tanh((x[:, 2] - 4) / 5) + 3 + (1 / 10) * z[:, 3]

    # Generate noise
    z = np.random.normal(loc=0.0, scale=1.0, size=(nsamples, 3))
    z_3 = np.random.laplace(loc=0.0, scale=1.0, size=(nsamples, 1))
    z = np.concatenate((z, z_3), axis=1)

    # Generate observational data
    x = generate_observational_data(z, dic_func)

    # Generate counterfactuals
    if num_interventions > 0:
        list_x_cf = []
        for _ in range(num_interventions):
            # Generate a new noise for each internvention
            z_f = np.random.normal(loc=0.0, scale=1.0, size=(num_samples_interventions, 3))
            z_f_3 = np.random.laplace(loc=0.0, scale=1.0, size=(num_samples_interventions, 1))
            z_f = np.concatenate((z_f, z_f_3), axis=1)
            x_cf = compute_counterfactual(z_f, dic_func)
            list_x_cf.append(x_cf)
    else:
        list_x_cf = None

    return x, z, adjacency_matrix, list_x_cf


def generate_symprod_simpson(
    nsamples: int, seed: int = 49, num_interventions: int = 4, num_samples_interventions: int = 100
) -> tuple[NDArray, NDArray, NDArray, list[NDArray] | None]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Generate graph
    adjacency_matrix = np.zeros((4, 4))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[1, 2] = 1
    adjacency_matrix[0, 2] = 1
    adjacency_matrix[0, 3] = 1
    adjacency_matrix = adjacency_matrix.T

    # Generate dic of functions
    dic_func = {}
    dic_func["f0"] = lambda x, z: z[:, 0]
    dic_func["f1"] = lambda x, z: 2 * np.tanh(2 * x[:, 0]) + np.sqrt(1 / 10) * z[:, 1]
    dic_func["f2"] = lambda x, z: (1 / 2) * x[:, 0] * x[:, 1] + (1 / 2) * z[:, 2]
    dic_func["f3"] = lambda x, z: 5 * np.tanh((3 / 2) * x[:, 0]) + 3 + np.sqrt(3 / 10) * z[:, 3]

    # Generate noise
    z_0 = np.random.normal(loc=0.0, scale=1.0, size=(nsamples, 1))
    z_1 = np.random.standard_t(df=3, size=(nsamples, 1))
    z_2 = np.random.laplace(loc=0.0, scale=1.0, size=(nsamples, 1))
    z_3 = np.random.normal(loc=0.0, scale=1.0, size=(nsamples, 1))
    z = np.concatenate((z_0, z_1, z_2, z_3), axis=1)

    # Generate observational data
    x = generate_observational_data(z, dic_func)

    # Generate counterfactuals
    if num_interventions > 0:
        list_x_cf = []
        for _ in range(num_interventions):
            # Generate a new noise for each internvention
            z_f_0 = np.random.normal(loc=0.0, scale=1.0, size=(num_samples_interventions, 1))
            z_f_1 = np.random.standard_t(df=3, size=(num_samples_interventions, 1))
            z_f_2 = np.random.laplace(loc=0.0, scale=1.0, size=(num_samples_interventions, 1))
            z_f_3 = np.random.normal(loc=0.0, scale=1.0, size=(num_samples_interventions, 1))
            z_f = np.concatenate((z_f_0, z_f_1, z_f_2, z_f_3), axis=1)
            x_cf = compute_counterfactual(z_f, dic_func)
            list_x_cf.append(x_cf)
    else:
        list_x_cf = None

    return x, z, adjacency_matrix, list_x_cf


def layer(parent: NDArray, noise: NDArray) -> NDArray:
    return np.log(1 + np.exp(1 + parent)) + np.log(1 + np.exp(0.5 + noise)) - 3.0


def generate_large_backdoor(
    nsamples: int, seed: int = 49, num_interventions: int = 9, num_samples_interventions: int = 100
) -> tuple[NDArray, NDArray, NDArray, list[NDArray] | None]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Generate graph
    adjacency_matrix = np.zeros((9, 9))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[0, 2] = 1
    adjacency_matrix[1, 3] = 1
    adjacency_matrix[2, 4] = 1
    adjacency_matrix[3, 5] = 1
    adjacency_matrix[4, 6] = 1
    adjacency_matrix[5, 7] = 1
    adjacency_matrix[6, 8] = 1
    adjacency_matrix[7, 8] = 1
    adjacency_matrix = adjacency_matrix.T

    # Generate dic of functions
    dic_func = {}
    dic_func["f0"] = lambda x, z: np.log(1 + np.exp(1.8 * z[:, 0])) - 1
    dic_func["f1"] = lambda x, z: layer(x[:, 0], np.zeros_like(x[:, 0])) * 1.5 + np.sqrt(0.25) * z[:, 1]
    dic_func["f2"] = lambda x, z: layer(x[:, 0], z[:, 2])
    dic_func["f3"] = lambda x, z: layer(x[:, 1], z[:, 3])
    dic_func["f4"] = lambda x, z: layer(x[:, 2], z[:, 4])
    dic_func["f5"] = lambda x, z: layer(x[:, 3], z[:, 5])
    dic_func["f6"] = lambda x, z: layer(x[:, 4], z[:, 6])
    dic_func["f7"] = lambda x, z: np.log(1 + np.exp(x[:, 5] + 1)) * 1.5 - 1 + np.sqrt(0.3) * z[:, 7]
    dic_func["f8"] = lambda x, z: -np.log(1 + np.exp(((-x[:, 6]) * 1.3 + x[:, 7]) / 3 + 1)) + 2 + 0.6 * z[:, 8]

    # Generate noise
    z = np.random.normal(loc=0.0, scale=1.0, size=(nsamples, 8))
    z_8 = np.random.laplace(loc=0.0, scale=1.0, size=(nsamples, 1))
    z = np.concatenate((z, z_8), axis=1)

    # Generate observational data
    x = generate_observational_data(z, dic_func)

    # Generate counterfactuals
    if num_interventions > 0:
        list_x_cf = []
        for _ in range(num_interventions):
            # Generate a new noise for each internvention
            z_f = np.random.normal(loc=0.0, scale=1.0, size=(num_samples_interventions, 8))
            z_f_8 = np.random.laplace(loc=0.0, scale=1.0, size=(num_samples_interventions, 1))
            z_f = np.concatenate((z_f, z_f_8), axis=1)
            x_cf = compute_counterfactual(z_f, dic_func)
            list_x_cf.append(x_cf)
    else:
        list_x_cf = None

    return x, z, adjacency_matrix, list_x_cf


def layerm(parent: NDArray, x_noise: NDArray) -> NDArray:
    """
    Implements soft truncation for both input and noise variables, Approximately preserves mean=0 and var=1.
    Reverses sign of input
    """
    return np.log(1 + np.exp(-parent + 1.5)) + np.log(1 + np.exp(0.5 + x_noise)) - 3


def generate_weak_arrow(
    nsamples: int, seed: int = 49, num_interventions: int = 9, num_samples_interventions: int = 100
) -> tuple[NDArray, NDArray, NDArray, list[NDArray] | None]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Generate graph
    adjacency_matrix = np.zeros((9, 9))
    adjacency_matrix[0, 1] = 1
    adjacency_matrix[0, 2] = 1
    adjacency_matrix[1, 3] = 1
    adjacency_matrix[2, 4] = 1
    adjacency_matrix[3, 5] = 1
    adjacency_matrix[4, 6] = 1
    adjacency_matrix[5, 7] = 1
    for i in range(8):
        adjacency_matrix[i, 8] = 1
    adjacency_matrix = adjacency_matrix.T

    # Generate dic of functions
    dic_func = {}
    dic_func["f0"] = lambda x, z: np.log(1 + np.exp(1.8 * z[:, 0])) - 1
    dic_func["f1"] = lambda x, z: layer(x[:, 0], np.zeros_like(x[:, 0])) * 0.75 + np.sqrt(0.75) * z[:, 1]
    dic_func["f2"] = lambda x, z: layerm(x[:, 0], z[:, 2])
    dic_func["f3"] = lambda x, z: layerm(x[:, 1], z[:, 3])
    dic_func["f4"] = lambda x, z: layer(x[:, 2], z[:, 4])
    dic_func["f5"] = lambda x, z: layer(x[:, 3], z[:, 5])
    dic_func["f6"] = lambda x, z: layer(x[:, 4], z[:, 6])
    dic_func["f7"] = lambda x, z: np.log(1 + np.exp(x[:, 5] + 1)) * 1.5 - 1 + np.sqrt(0.3) * z[:, 7]
    dic_func["f8"] = (
        lambda x, z: np.log(
            1
            + np.exp(
                0.1 * x[:, 0]
                + 0.1 * x[:, 1]
                + 0.1 * x[:, 2]
                + 0.1 * x[:, 3]
                + 0.1 * x[:, 4]
                + 0.1 * x[:, 5]
                + 0.5 * x[:, 6]
                + 0.7 * x[:, 7]
                + 1
            )
        )
        - 2
        + 0.5 * z[:, 8]
    )
    # Generate noise
    z = np.random.normal(loc=0.0, scale=1.0, size=(nsamples, 8))
    z_8 = np.random.laplace(loc=0.0, scale=1.0, size=(nsamples, 1))
    z = np.concatenate((z, z_8), axis=1)

    # Generate observational data
    x = generate_observational_data(z, dic_func)

    # Generate counterfactuals
    if num_interventions > 0:
        list_x_cf = []
        for _ in range(num_interventions):
            # Generate a new noise for each internvention
            z_f = np.random.normal(loc=0.0, scale=1.0, size=(num_samples_interventions, 8))
            z_f_8 = np.random.laplace(loc=0.0, scale=1.0, size=(num_samples_interventions, 1))
            z_f = np.concatenate((z_f, z_f_8), axis=1)
            x_cf = compute_counterfactual(z_f, dic_func)
            list_x_cf.append(x_cf)
    else:
        list_x_cf = None

    return x, z, adjacency_matrix, list_x_cf


def generate_data(
    seed_tr: int = 10,
    seed_te: int = 20,
    num_samples: int = 10000,
    data_type: CsuiteEnum = CsuiteEnum.LINGAUSS,
) -> tuple[NDArray, NDArray, NDArray, list[NDArray] | None, NDArray, NDArray]:
    match data_type:
        case CsuiteEnum.LINGAUSS:
            x_train, z_train, graph, x_cf_tr = generate_lingauss(num_samples, seed=seed_tr)
            x_test, z_test, graph, _ = generate_lingauss(num_samples, seed=seed_te)
        case CsuiteEnum.LINEXP:
            x_train, z_train, graph, x_cf_tr = generate_linexp(num_samples, seed=seed_tr)
            x_test, z_test, graph, _ = generate_linexp(num_samples, seed=seed_te)
        case CsuiteEnum.NONLINGAUSS:
            x_train, z_train, graph, x_cf_tr = generate_nonlingauss(num_samples, seed=seed_tr)
            x_test, z_test, graph, _ = generate_nonlingauss(num_samples, seed=seed_te)
        case CsuiteEnum.NONLIN_SIMPSON:
            x_train, z_train, graph, x_cf_tr = generate_nonlin_simpson(num_samples, seed=seed_tr)
            x_test, z_test, graph, _ = generate_nonlin_simpson(num_samples, seed=seed_te)
        case CsuiteEnum.SYMPROD_SIMPSON:
            x_train, z_train, graph, x_cf_tr = generate_symprod_simpson(num_samples, seed=seed_tr)
            x_test, z_test, graph, _ = generate_symprod_simpson(num_samples, seed=seed_te)
        case CsuiteEnum.LARGE_BACKDOOR:
            x_train, z_train, graph, x_cf_tr = generate_large_backdoor(num_samples, seed=seed_tr)
            x_test, z_test, graph, _ = generate_large_backdoor(num_samples, seed=seed_te)
        case CsuiteEnum.WEAK_ARROW:
            x_train, z_train, graph, x_cf_tr = generate_weak_arrow(num_samples, seed=seed_tr)
            x_test, z_test, graph, _ = generate_weak_arrow(num_samples, seed=seed_te)
        case _:
            raise NotImplementedError("Distribution case not supported")

    return x_train, z_train, graph, x_cf_tr, x_test, z_test


def main(seed: int, val_train_split_ratio: float, dist_case: CsuiteEnum, dataset_size: int):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Generate the dataset
    X_train, Z_train, graph, list_x_cf, X_test, Z_test = generate_data(
        seed_tr=seed,
        seed_te=10 + seed,
        num_samples=dataset_size,
        data_type=dist_case,
    )

    data_train = np.concatenate((X_train, Z_train), axis=1)
    data_test = np.concatenate((X_test, Z_test), axis=1)

    total_nodes = X_train.shape[-1]
    base_dir = os.path.join(
        "src",
        "fip",
        "data",
        dist_case.value,
        "total_nodes_" + str(total_nodes),
        "seed_" + str(seed),
    )
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # save the true graph
    f = os.path.join(base_dir, "true_graph.npy")
    np.save(f, graph)

    # Save the corresponding datasets
    val_dataset_size = int(val_train_split_ratio * dataset_size)
    data = data_train[val_dataset_size:]
    f = os.path.join(base_dir, "train" + "_" + "x" + ".npy")
    np.save(f, data)
    print("Save training data")

    data = data_train[:val_dataset_size]
    f = os.path.join(base_dir, "val" + "_" + "x" + ".npy")
    np.save(f, data)
    print("Save validation data")

    data = data_test
    f = os.path.join(base_dir, "test" + "_" + "x" + ".npy")
    np.save(f, data)
    print("Save testing data")

    if list_x_cf is not None:
        for k, data in enumerate(list_x_cf):
            f = os.path.join(base_dir, "x_cf_" + str(k) + ".npy")
            np.save(f, data)
            print("Save counterfactual data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val_train_split_ratio", type=float, default=0.25)
    parser.add_argument("--dist_case", type=CsuiteEnum, help="Distribution case", choices=CsuiteEnum, required=True)
    parser.add_argument("--dataset_size", type=int, default=10000)
    args = parser.parse_args()

    main(
        seed=args.seed,
        val_train_split_ratio=args.val_train_split_ratio,
        dist_case=args.dist_case,
        dataset_size=args.dataset_size,
    )
