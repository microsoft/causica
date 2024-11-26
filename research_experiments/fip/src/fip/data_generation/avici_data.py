import argparse
import os
import random
from enum import Enum

import numpy as np
import torch

from causica.distributions.transforms import TensorToTensorDictTransform
from causica.sem.distribution_parameters_sem import DistributionParametersSEM
from fip.data_generation.config_data import LinearConfig, RFFConfig, SFConfig
from fip.data_generation.sem_factory import SemSamplerFactory
from fip.data_modules.synthetic_data_module import sample_counterfactual


class NoiseEnum(Enum):
    """Different type of noise."""

    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"
    CAUCHY = "cauchy"


class GraphEnum(Enum):
    """Different type of graphs."""

    ER = "er"
    SF_IN = "sf_in"
    SF_OUT = "sf_out"
    SBM = "sbm"
    GRG = "grg"
    WS = "ws"


class FuncEnum(Enum):
    """Different type of functional relationships."""

    LINEAR = "linear"
    RFF = "rff"


class DistEnum(Enum):
    """Different type of distribution cases."""

    IN = "in"
    OUT = "out"


def gen_from_factory_sem(
    data_dim: int,
    func_type: FuncEnum,
    noise_type: NoiseEnum,
    graph_type: GraphEnum,
    dist_case: DistEnum,
) -> DistributionParametersSEM:

    match dist_case:
        case DistEnum.IN:
            sem_sampler = SemSamplerFactory(
                node_nums=[data_dim], noises=[noise_type.value], funcs=[func_type.value], graphs=[graph_type.value]
            )
            sem = sem_sampler()[0].sample()
        case DistEnum.OUT:
            config_linear = LinearConfig()
            config_linear.weight_low = 0.5
            config_linear.weight_high = 4.0

            config_rff = RFFConfig()
            config_rff.length_low = 5.0
            config_rff.length_high = 12.0
            config_rff.out_low = 8.0
            config_rff.out_high = 22.0

            match graph_type:
                case GraphEnum.SF_OUT:
                    config_sf = SFConfig()
                    config_sf.edges_per_node = [2]
                    config_sf.attach_power = [0.5, 1.5]
                case _:
                    config_sf = SFConfig()

            sem_sampler = SemSamplerFactory(
                node_nums=[data_dim],
                noises=[noise_type.value],
                funcs=[func_type.value],
                graphs=[graph_type.value],
                config_rff=config_rff,
                config_linear=config_linear,
                config_sf=config_sf,
            )
            sem = sem_sampler()[0].sample()

    return sem


def main(
    data_dir: str,
    seed: int,
    val_train_split_ratio: float,
    train_size: int,
    test_size: int,
    data_dim: int,
    func_type: FuncEnum,
    noise_type: NoiseEnum,
    graph_type: GraphEnum,
    dist_case: DistEnum,
    num_interventions: int = 1,
    num_intervention_samples: int = 100,
):
    # Random Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    name_data = f"{graph_type.value}_{func_type.value}_{noise_type.value}_{dist_case.value}"
    base_dir = os.path.join(
        data_dir,
        name_data,
        "total_nodes_" + str(data_dim),
        "seed_" + str(seed),
    )

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    sem = gen_from_factory_sem(
        data_dim=data_dim,
        func_type=func_type,
        noise_type=noise_type,
        graph_type=graph_type,
        dist_case=dist_case,
    )

    w = np.transpose(sem.graph.numpy())
    f = os.path.join(base_dir, "true_graph.npy")
    np.save(f, w)

    shapes_dict = {f"x_{i}": torch.Size([1]) for i in range(data_dim)}
    td_to_tensor_transform = TensorToTensorDictTransform(shapes_dict)

    for data_split in ["train", "val", "test"]:

        print("Data Case: ", data_split)
        if data_split == "train":
            dataset_size = train_size
        if data_split == "val":
            dataset_size = int(val_train_split_ratio * train_size)
        elif data_split == "test":
            dataset_size = test_size

        # Sample data and noise from the SEM
        noise = sem.sample_noise(torch.Size([dataset_size]))
        data = sem.noise_to_sample(noise)

        # from tensordict to numpy array
        data_np = td_to_tensor_transform.inv(data).numpy()
        noise_np = td_to_tensor_transform.inv(noise).numpy()

        x = np.concatenate((data_np, noise_np), axis=1)

        # Save the corresponding dataset
        f = os.path.join(base_dir, data_split + "_" + "x" + ".npy")
        np.save(f, x)

    list_cf = []
    noise_cf = sem.sample_noise(torch.Size([num_intervention_samples]))
    data_cf = sem.noise_to_sample(noise_cf)
    for i in range(num_interventions):
        list_cf.append(sample_counterfactual(sem, data_cf, noise_cf))

    # Save the corresponding counterfactual dataset
    for i in range(num_interventions):
        f_data = list_cf[i].factual_data
        n_data = list_cf[i].noise_data
        cf_data = list_cf[i].counterfactual_data
        int_values = list_cf[i].intervention_values

        f_data_np = td_to_tensor_transform.inv(f_data).numpy()
        n_data_np = td_to_tensor_transform.inv(n_data).numpy()
        cf_data_np = td_to_tensor_transform.inv(cf_data).numpy()

        list_keys = list(int_values.keys())
        assert len(list_keys) == 1
        int_index = int(list_keys[0].split("_")[1])
        int_index_np = np.ones((num_intervention_samples, 1)) * int_index
        int_values_np = torch.cat(list(int_values.values()), dim=-1).repeat(num_intervention_samples, 1).numpy()

        x_cf = np.concatenate((f_data_np, n_data_np, cf_data_np, int_index_np, int_values_np), axis=1)
        f = os.path.join(base_dir, "x_cf_" + str(i) + ".npy")
        np.save(f, x_cf)


if __name__ == "__main__":
    # Input Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for initialization")
    parser.add_argument(
        "--val_train_split_ratio",
        type=float,
        default=0.25,
        help="Pecentage of samples in validation set as comapred to the training set",
    )
    parser.add_argument("--train_size", type=int, default=10000, help="Size of the train split")
    parser.add_argument("--test_size", type=int, default=10000, help="Size of the test split")
    parser.add_argument("--data_dim", type=int, default=11, help="Dimension of the data")

    parser.add_argument(
        "--func_type",
        type=FuncEnum,
        help="Type of functional relationship",
        choices=FuncEnum,
        required=True,
    )
    parser.add_argument(
        "--noise_type", type=NoiseEnum, help="Type of noise distribution", choices=NoiseEnum, required=True
    )

    parser.add_argument("--graph_type", type=GraphEnum, help="Type of graph", choices=GraphEnum, required=True)

    parser.add_argument("--dist_case", type=DistEnum, help="Distribution case", choices=DistEnum, required=True)

    parser.add_argument("--num_interventions", type=int, default=1, help="Number of interventions")
    parser.add_argument(
        "--num_intervention_samples", type=int, default=100, help="Number of samples for each intervention"
    )

    args = parser.parse_args()
    main(
        args.data_dir,
        args.seed,
        args.val_train_split_ratio,
        args.train_size,
        args.test_size,
        args.data_dim,
        args.func_type,
        args.noise_type,
        args.graph_type,
        args.dist_case,
        args.num_interventions,
        args.num_intervention_samples,
    )
