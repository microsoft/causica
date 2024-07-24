import argparse
import math
import os
import random
from enum import Enum

import numpy as np
import torch
import torch.distributions as td
from fip.data_utils.synthetic_data_module import sample_counterfactual

from causica.data_generation.samplers.functional_relationships_sampler import (
    FunctionalRelationshipsSampler,
    HeteroscedasticRFFFunctionalRelationshipsSampler,
    LinearRelationshipsSampler,
    RFFFunctionalRelationshipsSampler,
)
from causica.data_generation.samplers.noise_dist_sampler import (
    JointNoiseModuleSampler,
    NoiseModuleSampler,
    UnivariateCauchyNoiseModuleSampler,
    UnivariateLaplaceNoiseModuleSampler,
    UnivariateNormalNoiseModuleSampler,
)
from causica.data_generation.samplers.sem_sampler import SEMSampler
from causica.distributions.adjacency import (
    AdjacencyDistribution,
    EdgesPerNodeErdosRenyiDAGDistribution,
    GeometricRandomGraphDAGDistribution,
    ScaleFreeDAGDistribution,
    StochasticBlockModelDAGDistribution,
    WattsStrogatzDAGDistribution,
)
from causica.distributions.signed_uniform import MultivariateSignedUniform
from causica.distributions.transforms import TensorToTensorDictTransform
from causica.sem.distribution_parameters_sem import DistributionParametersSEM


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


def gen_sem(
    data_dim: int,
    func_type: FuncEnum,
    noise_type: NoiseEnum,
    graph_type: GraphEnum,
    dist_case: DistEnum,
) -> DistributionParametersSEM:
    """Function for generating structural causal model with a random DAG sampled using erdos renyi scheme and support over different noise/functional distributions.

    Args:
        data_dim: Dimension of data points
        func_type: Type of functional relationship
        noise_type: Type of noise distribution
        graph_type: Type of graph
        dist_case: in Distribution or out Distribution

    Returns:
        Structural Causal Model: Object of class DistributionParametersSEM
    """
    shapes_dict = {f"x_{i}": torch.Size([1]) for i in range(data_dim)}
    dim = sum(shape.numel() for shape in shapes_dict.values())

    std_dist: td.Distribution | float
    noise_dist_samplers: dict[str, NoiseModuleSampler]
    adjacency_dist: AdjacencyDistribution
    functional_relationships_sampler: FunctionalRelationshipsSampler
    match dist_case:
        case DistEnum.IN:
            log_functional_rescaling_sampler = None
        case DistEnum.OUT:
            if func_type == FuncEnum.LINEAR and noise_type == NoiseEnum.GAUSSIAN:
                # in case of LGM, we do not apply the data-dependent transformation of the variance for the noise
                log_functional_rescaling_sampler = None
            else:
                num_rf = 100
                zeros_vector_rf = torch.zeros((num_rf,), dtype=torch.float32)
                ones_vector_rf = torch.ones((num_rf,), dtype=torch.float32)
                zeros_matrix = torch.zeros((num_rf, dim), dtype=torch.float32)
                ones_vector_dim = torch.ones((dim,), dtype=torch.float32)
                log_functional_rescaling_sampler = HeteroscedasticRFFFunctionalRelationshipsSampler(
                    rf_dist=td.MultivariateNormal(zeros_matrix, covariance_matrix=torch.eye(dim)),
                    coeff_dist=td.MultivariateNormal(zeros_vector_rf, covariance_matrix=torch.eye(num_rf)),
                    shapes_dict=shapes_dict,
                    length_dist=10.0,
                    out_dist=2.0,
                    angle_dist=td.Uniform(low=zeros_vector_rf, high=2 * math.pi * ones_vector_rf),
                    log_scale=True,
                )

    match noise_type:
        case NoiseEnum.GAUSSIAN:
            match dist_case:
                case DistEnum.IN:
                    if func_type == FuncEnum.LINEAR:
                        # in case of LGM, we fix the variance of all the noise to be identifiable
                        std_dist = td.Uniform(low=0.2, high=2.0).sample().item()
                    else:
                        std_dist = td.Uniform(low=0.2, high=2.0)
                case DistEnum.OUT:
                    std_dist = 1.0
                case _:
                    raise NotImplementedError("Distribution case not supported")
            noise_dist_samplers = {
                f"x_{i}": UnivariateNormalNoiseModuleSampler(std_dist=std_dist, dim=1) for i in range(data_dim)
            }
        case NoiseEnum.LAPLACE:
            match dist_case:
                case DistEnum.IN:
                    std_dist = td.Uniform(low=0.2, high=2.0)
                case DistEnum.OUT:
                    std_dist = 1.0
                case _:
                    raise NotImplementedError("Distribution case not supported")
            noise_dist_samplers = {
                f"x_{i}": UnivariateLaplaceNoiseModuleSampler(std_dist=std_dist, dim=1) for i in range(data_dim)
            }
        case NoiseEnum.CAUCHY:
            match dist_case:
                case DistEnum.IN:
                    std_dist = td.Uniform(low=0.2, high=2.0)
                case DistEnum.OUT:
                    std_dist = 1.0
                case _:
                    raise NotImplementedError("Distribution case not supported")
            noise_dist_samplers = {
                f"x_{i}": UnivariateCauchyNoiseModuleSampler(std_dist=std_dist, dim=1) for i in range(data_dim)
            }
        case _:
            raise NotImplementedError("Noise distribution not supported")

    joint_noise_module_sampler = JointNoiseModuleSampler(noise_dist_samplers)

    match graph_type:
        case GraphEnum.ER:
            adjacency_dist = EdgesPerNodeErdosRenyiDAGDistribution(num_nodes=data_dim, edges_per_node=[1, 2, 3])
        case GraphEnum.SF_IN:
            adjacency_dist = ScaleFreeDAGDistribution(
                num_nodes=data_dim, edges_per_node=[1, 2, 3], power=[1.0], in_degree=True
            )
        case GraphEnum.SF_OUT:
            match dist_case:
                case DistEnum.IN:
                    edges_per_node = [1, 2, 3]
                    power = [1.0]
                case DistEnum.OUT:
                    edges_per_node = [2]
                    power = [0.5, 1.5]
                case _:
                    raise NotImplementedError("Distribution case not supported")
            adjacency_dist = ScaleFreeDAGDistribution(
                num_nodes=data_dim, edges_per_node=edges_per_node, power=power, in_degree=False
            )
        case GraphEnum.SBM:
            adjacency_dist = StochasticBlockModelDAGDistribution(
                num_nodes=data_dim, edges_per_node=[1, 2, 3], num_blocks=[5, 10], damping=[0.1]
            )
        case GraphEnum.GRG:
            adjacency_dist = GeometricRandomGraphDAGDistribution(num_nodes=data_dim, radius=[0.1])
        case GraphEnum.WS:
            adjacency_dist = WattsStrogatzDAGDistribution(
                num_nodes=data_dim, lattice_dim=[2, 3], rewire_prob=[0.3], neighbors=[1]
            )
        case _:
            raise NotImplementedError("Graph type not supported")

    match func_type:
        case FuncEnum.LINEAR:
            match dist_case:
                case DistEnum.IN:
                    low = 1.0
                    high = 3.0
                case DistEnum.OUT:
                    low = 0.5
                    high = 4.0
                case _:
                    raise NotImplementedError("Distribution case not supported")

            one_vector_dim = torch.ones((dim,), dtype=torch.float32)
            functional_relationships_sampler = LinearRelationshipsSampler(
                scale_dist=MultivariateSignedUniform(
                    low=low,
                    high=high,
                    size=torch.Size([dim, dim]),
                ),
                shapes_dict=shapes_dict,
                bias_dist=td.Uniform(
                    low=-3.0 * one_vector_dim,
                    high=3.0 * one_vector_dim,
                ),
            )
        case FuncEnum.RFF:
            num_rf = 100
            match dist_case:
                case DistEnum.IN:
                    low_l = 7.0
                    high_l = 10.0
                    low_c = 10.0
                    high_c = 20.0
                case DistEnum.OUT:
                    low_l = 5.0
                    high_l = 12.0
                    low_c = 8.0
                    high_c = 22.0
                case _:
                    raise NotImplementedError("Distribution case not supported")
            zeros_vector_rf = torch.zeros((num_rf,), dtype=torch.float32)
            ones_vector_rf = torch.ones((num_rf,), dtype=torch.float32)
            zeros_matrix = torch.zeros((num_rf, dim), dtype=torch.float32)
            ones_vector_dim = torch.ones((dim,), dtype=torch.float32)
            functional_relationships_sampler = RFFFunctionalRelationshipsSampler(
                rf_dist=td.MultivariateNormal(zeros_matrix, covariance_matrix=torch.eye(dim)),
                coeff_dist=td.MultivariateNormal(zeros_vector_rf, covariance_matrix=torch.eye(num_rf)),
                shapes_dict=shapes_dict,
                bias_dist=td.Uniform(
                    low=-3.0 * ones_vector_dim,
                    high=3.0 * ones_vector_dim,
                ),
                length_dist=td.Uniform(
                    low=low_l * ones_vector_dim,
                    high=high_l * ones_vector_dim,
                ),
                out_dist=td.Uniform(
                    low=low_c * ones_vector_dim,
                    high=high_c * ones_vector_dim,
                ),
                angle_dist=td.Uniform(low=zeros_vector_rf, high=2 * math.pi * ones_vector_rf),
            )
        case _:
            raise NotImplementedError("Functional relationship type not supported")

    sem = SEMSampler(
        adjacency_dist=adjacency_dist,
        joint_noise_module_sampler=joint_noise_module_sampler,
        functional_relationships_sampler=functional_relationships_sampler,
        log_functional_rescaling_sampler=log_functional_rescaling_sampler,
    ).sample()

    return sem


def main(
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
        "src",
        "fip",
        "data",
        name_data,
        "total_nodes_" + str(data_dim),
        "seed_" + str(seed),
    )

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    sem = gen_sem(
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
        cf_data = list_cf[i].counterfactual_data
        int_values = list_cf[i].intervention_values

        f_data_np = td_to_tensor_transform.inv(f_data).numpy()
        cf_data_np = td_to_tensor_transform.inv(cf_data).numpy()

        list_keys = list(int_values.keys())
        assert len(list_keys) == 1
        int_index = int(list_keys[0].split("_")[1])
        int_index_np = np.ones((num_intervention_samples, 1)) * int_index
        int_values_np = torch.cat(list(int_values.values()), dim=-1).repeat(num_intervention_samples, 1).numpy()

        x_cf = np.concatenate((f_data_np, cf_data_np, int_index_np, int_values_np), axis=1)
        f = os.path.join(base_dir, "x_cf_" + str(i) + ".npy")
        np.save(f, x_cf)


if __name__ == "__main__":
    # Input Parsing
    parser = argparse.ArgumentParser()
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
