import argparse
import os
import random
from enum import Enum
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch.distributions import Distribution, Independent, Normal, Uniform


class NFEnum(Enum):
    """Different data generation processes."""

    TRIANGLE = "triangle"
    SYMPROD_SIMPSON = "symprod_simpson"
    LARGE_BACKDOOR = "large_backdoor"


pu_dict: dict[int, Callable] = {}


def base_distribution_3_nodes() -> Distribution:

    p_u = Independent(
        Normal(
            torch.zeros(3),
            torch.ones(3),
        ),
        1,
    )
    return p_u


def base_distribution_4_nodes() -> Distribution:
    p_u = Independent(
        Normal(
            torch.zeros(4),
            torch.ones(4),
        ),
        1,
    )
    return p_u


def base_distribution_9_nodes() -> Distribution:
    p_u = Independent(
        Uniform(
            1e-6,
            torch.ones(9),
        ),
        1,
    )
    return p_u


pu_dict[3] = base_distribution_3_nodes
pu_dict[4] = base_distribution_4_nodes
pu_dict[9] = base_distribution_9_nodes


def generate_observational_data(noise: NDArray, dic_func: dict[str, Callable]) -> NDArray:
    x = np.zeros(noise.shape)
    for k in range(len(dic_func)):
        x[:, k] = dic_func[f"f{k}"](x, noise)
    return x


def generate_observational_data_torch(noise: torch.Tensor, dic_func: dict[str, Callable]) -> torch.Tensor:
    x = torch.zeros(noise.shape)
    for k in range(len(dic_func)):
        x[:, k] = dic_func[f"f{k}"](x, noise)
    return x


def get_intervention_list(x_train: NDArray, intervention_index_list: list[int]) -> list[dict[str, str | int | float]]:
    perc_idx = [25, 50, 75]

    percentiles = np.percentile(x_train, perc_idx, axis=0)
    int_list = []
    for i in intervention_index_list:
        percentiles_i = percentiles[:, i]
        values_i = []
        for perc_name, perc_value in zip(perc_idx, percentiles_i):
            values_i.append({"name": f"{perc_name}p", "value": perc_value})

        for value in values_i:
            value["value"] = round(value["value"], 2)
            value["index"] = i
            int_list.append(value)

    return int_list


def compute_counterfactual(
    noise: NDArray, dic_func: dict[str, Callable], intervention_dic: dict[str, str | int | float]
) -> NDArray:
    # generate factual data
    x_f = generate_observational_data(noise, dic_func)

    val_int = intervention_dic["value"]
    idx_int = intervention_dic["index"]

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


def compute_counterfactual_torch(
    noise: torch.Tensor, dic_func: dict[str, Callable], intervention_dic: dict[str, str | int | float]
) -> NDArray:
    # generate factual data
    x_f = generate_observational_data_torch(noise, dic_func)

    val_int = intervention_dic["value"]
    assert isinstance(val_int, float), "val_int should be a float"
    idx_int = intervention_dic["index"]
    assert isinstance(idx_int, int), "idx_int should be an integer"

    # create the interventional function
    dic_func_cf = dic_func.copy()
    dic_func_cf[f"f{idx_int}"] = lambda _, z: z[:, idx_int]

    # create the counterfactual noise
    noise_cf = noise.clone()
    noise_cf[:, idx_int] = val_int

    # compute the counterfactual
    x_cf = generate_observational_data_torch(noise_cf, dic_func_cf)

    # concatenate all the data
    x_all = np.concatenate((x_f, x_cf), axis=1)
    x_all = np.concatenate((x_all, np.array([idx_int] * len(x_all)).reshape(-1, 1)), axis=1)
    x_all = np.concatenate((x_all, np.array([val_int] * len(x_all)).reshape(-1, 1)), axis=1)

    return x_all


def generate_triangle(
    num_samples: int, seed: int = 49, num_samples_intervention: int = 1000
) -> tuple[NDArray, NDArray, NDArray, list[NDArray]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    intervention_index_list = [0, 1]

    adj = torch.zeros((3, 3))
    adj[0, :] = torch.tensor([0, 0, 0])
    adj[1, :] = torch.tensor([1, 0, 0])
    adj[2, :] = torch.tensor([1, 1, 0])
    adj_np: NDArray = adj.numpy()

    # Generate dic of functions
    dic_func = {}
    dic_func["f0"] = lambda x, z: z[:, 0] + 1.0
    dic_func["f1"] = lambda x, z: 10 * x[:, 0] - z[:, 1]
    dic_func["f2"] = lambda x, z: 0.5 * x[:, 1] + x[:, 0] + 1.0 * z[:, 2]

    # Generate noise
    z = pu_dict[3]().sample((num_samples,))
    assert isinstance(z, torch.Tensor), "z should be a torch.Tensor"
    z_np: NDArray = z.numpy()

    # Generate observational data
    x = generate_observational_data(z_np, dic_func)

    # Generate interventional data
    int_list = get_intervention_list(x, intervention_index_list)
    list_x_cf = []
    for intervention_dic in int_list:
        # Generate a new noise for each intervention
        z_f = pu_dict[3]().sample((num_samples_intervention,))
        z_f = z_f.numpy()
        x_cf = compute_counterfactual(z_f, dic_func, intervention_dic)
        list_x_cf.append(x_cf)

    return x, z_np, adj_np, list_x_cf


def inv_softplus(bias: torch.Tensor) -> torch.Tensor:
    return bias.expm1().clamp_min(1e-6).log()


def layer(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.softplus(x + 1) + F.softplus(0.5 + y) - 3.0


def inv_layer(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return inv_softplus(z + 3 - F.softplus(x + 1)) - 0.5


def icdf_laplace(loc: torch.Tensor, scale: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    term = value - 0.5
    return loc - scale * term.sign() * torch.log1p(-2 * term.abs())


def cdf_laplace(loc: torch.Tensor, scale: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return 0.5 - 0.5 * (value - loc).sign() * torch.expm1(-(value - loc).abs() / scale)


def generate_large_backdoor(
    num_samples: int, seed: int = 49, num_samples_intervention: int = 1000
) -> tuple[NDArray, NDArray, NDArray, list[NDArray]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    intervention_index_list = [0, 1, 2, 4]

    adj = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0],
        ]
    )
    adj_np: NDArray = adj.numpy()

    # Generate dic of functions
    dic_func = {}
    dic_func["f0"] = lambda x, z: F.softplus(1.8 * z[:, 0]) - 1
    dic_func["f1"] = lambda x, z: 0.25 * z[:, 1] + layer(x[:, 0], torch.zeros_like(z[:, 1])) * 1.5
    dic_func["f2"] = lambda x, z: layer(x[:, 0], z[:, 2])
    dic_func["f3"] = lambda x, z: layer(x[:, 1], z[:, 3])
    dic_func["f4"] = lambda x, z: layer(x[:, 2], z[:, 4])
    dic_func["f5"] = lambda x, z: layer(x[:, 3], z[:, 5])
    dic_func["f6"] = lambda x, z: layer(x[:, 4], z[:, 6])
    dic_func["f7"] = lambda x, z: 0.3 * z[:, 7] + (F.softplus(x[:, 5] + 1) - 1)
    dic_func["f8"] = lambda x, z: icdf_laplace(
        -F.softplus((x[:, 6] * 1.3 + x[:, 7]) / 3 + 1) + 2, torch.tensor([0.6]), z[:, 8]
    )

    # Generate noise
    z_torch = pu_dict[9]().sample((num_samples,))
    assert isinstance(z_torch, torch.Tensor), "z_torch should be a torch.Tensor"
    z_np: NDArray = z_torch.numpy()

    # Generate observational data
    x_torch = generate_observational_data_torch(z_torch, dic_func)
    x: NDArray = x_torch.numpy()

    # Generate interventional data
    int_list = get_intervention_list(x, intervention_index_list)
    list_x_cf = []
    for intervention_dic in int_list:
        # Generate a new noise for each intervention
        z_f_torch = pu_dict[9]().sample((num_samples_intervention,))
        x_cf = compute_counterfactual_torch(z_f_torch, dic_func, intervention_dic)
        list_x_cf.append(x_cf)

    return x, z_np, adj_np, list_x_cf


def generate_symprod_simpson(
    num_samples: int, seed: int = 49, num_samples_intervention: int = 1000
) -> tuple[NDArray, NDArray, NDArray, list[NDArray]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    intervention_index_list = [0, 1, 2]

    adj = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ]
    )
    adj_np: NDArray = adj.numpy()

    # Generate dic of functions
    dic_func = {}
    dic_func["f0"] = lambda x, z: z[:, 0]
    dic_func["f1"] = lambda x, z: 2 * torch.tanh(2 * x[:, 0]) + 1 / np.sqrt(10) * z[:, 1]
    dic_func["f2"] = lambda x, z: 1 / 2 * x[:, 0] * x[:, 1] + 1 / np.sqrt(2) * z[:, 2]
    dic_func["f3"] = lambda x, z: torch.tanh(3 / 2 * x[:, 0]) + np.sqrt(3 / 10) * z[:, 3]

    # Generate noise
    z_torch = pu_dict[4]().sample((num_samples,))
    assert isinstance(z_torch, torch.Tensor), "z_torch should be a torch.Tensor"
    z_np: NDArray = z_torch.numpy()

    # Generate observational data
    x_torch = generate_observational_data_torch(z_torch, dic_func)
    x_np: NDArray = x_torch.numpy()

    # Generate interventional data
    int_list = get_intervention_list(x_np, intervention_index_list)
    list_x_cf = []
    for intervention_dic in int_list:
        # Generate a new noise for each intervention
        z_f_torch = pu_dict[4]().sample((num_samples_intervention,))
        x_cf = compute_counterfactual_torch(z_f_torch, dic_func, intervention_dic)
        list_x_cf.append(x_cf)

    return x_np, z_np, adj_np, list_x_cf


def generate_data(
    seed_tr: int = 10,
    seed_te: int = 20,
    num_samples: int = 10000,
    data_type: NFEnum = NFEnum.TRIANGLE,
) -> tuple[NDArray, NDArray, NDArray, list[NDArray], NDArray, NDArray]:
    match data_type:
        case NFEnum.TRIANGLE:
            x_train, z_train, graph, x_cf_tr = generate_triangle(num_samples, seed=seed_tr)
            x_test, z_test, graph, _ = generate_triangle(num_samples, seed=seed_te)
        case NFEnum.SYMPROD_SIMPSON:
            x_train, z_train, graph, x_cf_tr = generate_symprod_simpson(num_samples, seed=seed_tr)
            x_test, z_test, graph, _ = generate_symprod_simpson(num_samples, seed=seed_te)
        case NFEnum.LARGE_BACKDOOR:
            x_train, z_train, graph, x_cf_tr = generate_large_backdoor(num_samples, seed=seed_tr)
            x_test, z_test, graph, _ = generate_large_backdoor(num_samples, seed=seed_te)
        case _:
            raise NotImplementedError("Distribution case not supported")

    return x_train, z_train, graph, x_cf_tr, x_test, z_test


def main(seed: int, val_train_split_ratio: float, dist_case: NFEnum, dataset_size: int):
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
        ".",
        "data",
        "nf_" + dist_case.value,
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

    for k, data in enumerate(list_x_cf):
        f = os.path.join(base_dir, "x_cf_" + str(k) + ".npy")
        np.save(f, data)
        print("Save counterfactual data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val_train_split_ratio", type=float, default=0.25)
    parser.add_argument("--dist_case", type=NFEnum, help="Distribution case", choices=NFEnum, required=True)
    parser.add_argument("--dataset_size", type=int, default=10000)
    args = parser.parse_args()

    main(
        seed=args.seed,
        val_train_split_ratio=args.val_train_split_ratio,
        dist_case=args.dist_case,
        dataset_size=args.dataset_size,
    )
