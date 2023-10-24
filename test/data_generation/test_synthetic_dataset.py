import torch
import torch.distributions as td
from torch.utils.data import DataLoader

from causica.data_generation.samplers.functional_relationships_sampler import LinearRelationshipsSampler
from causica.data_generation.samplers.noise_dist_sampler import (
    BernoulliNoiseModuleSampler,
    JointNoiseModuleSampler,
    UnivariateNormalNoiseModuleSampler,
)
from causica.data_generation.samplers.sem_sampler import SEMSampler
from causica.data_generation.synthetic_dataset import CausalDataset
from causica.datasets.interventional_data import InterventionData
from causica.distributions import ErdosRenyiDAGDistribution
from causica.lightning.data_modules.synthetic_data_module import _tuple_collate_fn


def test_mixed_type_causal_dataset():

    shapes_dict = {
        "x_0": torch.Size([1]),
        "x_1": torch.Size([5]),
        "x_2": torch.Size([1]),
    }

    noise_dist_samplers = {
        "x_0": UnivariateNormalNoiseModuleSampler(std_dist=td.Uniform(low=0.2, high=2.0), dim=1),
        "x_1": UnivariateNormalNoiseModuleSampler(std_dist=td.Uniform(low=0.2, high=2.0), dim=5),
        "x_2": BernoulliNoiseModuleSampler(base_logits_dist=td.Uniform(low=0.2, high=2.0), dim=1),
    }

    # Create adjacency distribution, joint noise module sampler, and functional relationships sampler
    adjacency_dist = ErdosRenyiDAGDistribution(num_nodes=3, probs=torch.tensor(0.2))
    joint_noise_module_sampler = JointNoiseModuleSampler(noise_dist_samplers)
    initial_linear_coefficient_matrix_shape = sum(shape[0] for shape in shapes_dict.values())
    functional_relationships_sampler = LinearRelationshipsSampler(
        td.Uniform(
            low=torch.ones((initial_linear_coefficient_matrix_shape, initial_linear_coefficient_matrix_shape)),
            high=3.0 * torch.ones((initial_linear_coefficient_matrix_shape, initial_linear_coefficient_matrix_shape)),
        ),
        shapes_dict,
    )

    sem_sampler = SEMSampler(adjacency_dist, joint_noise_module_sampler, functional_relationships_sampler)

    dataset = CausalDataset(sem_sampler, 5, 7, 1, 2)

    for sample in dataset:
        assert sample[0]["x_0"].shape == torch.Size([5, 1])
        assert sample[0].batch_size == torch.Size([5])
        assert sample[1]["x_0"].shape == torch.Size([5, 1])
        assert sample[1].batch_size == torch.Size([5])
        assert sample[0]["x_1"].shape == torch.Size([5, 5])
        assert sample[1]["x_1"].shape == torch.Size([5, 5])
        assert sample[2].shape == torch.Size([3, 3])
        assert len(sample[3]) == 1
        assert isinstance(sample[3][0], InterventionData)
        assert all((sample[0]["x_2"].unique()[:, None] == torch.tensor([0.0, 1.0])).any(dim=1))


def test_causal_dataset_dataloader():
    shapes_dict = {
        "x_0": torch.Size([1]),
        "x_1": torch.Size([1]),
        "x_2": torch.Size([1]),
    }

    noise_dist_samplers = {
        "x_0": UnivariateNormalNoiseModuleSampler(std_dist=td.Uniform(low=0.2, high=2.0), dim=1),
        "x_1": UnivariateNormalNoiseModuleSampler(std_dist=td.Uniform(low=0.2, high=2.0), dim=1),
        "x_2": UnivariateNormalNoiseModuleSampler(std_dist=td.Uniform(low=0.2, high=2.0), dim=1),
    }
    # Create adjacency distribution, joint noise module sampler, and functional relationships sampler
    adjacency_dist = ErdosRenyiDAGDistribution(num_nodes=3, probs=torch.tensor(0.2))
    joint_noise_module_sampler = JointNoiseModuleSampler(noise_dist_samplers)
    initial_linear_coefficient_matrix_shape = sum(shape[0] for shape in shapes_dict.values())
    functional_relationships_sampler = LinearRelationshipsSampler(
        td.Uniform(
            low=torch.ones((initial_linear_coefficient_matrix_shape, initial_linear_coefficient_matrix_shape)),
            high=3.0 * torch.ones((initial_linear_coefficient_matrix_shape, initial_linear_coefficient_matrix_shape)),
        ),
        shapes_dict,
    )

    sem_sampler = SEMSampler(adjacency_dist, joint_noise_module_sampler, functional_relationships_sampler)

    dataset = CausalDataset(sem_sampler, 5, 7, 1, 2)

    loader = DataLoader(dataset, batch_size=2, collate_fn=_tuple_collate_fn, drop_last=True)

    for sample in loader:
        assert sample[0]["x_0"].shape == torch.Size([2, 5, 1])
        assert sample[0].batch_size == torch.Size([2, 5])
        assert sample[1]["x_0"].shape == torch.Size([2, 5, 1])
        assert sample[1].batch_size == torch.Size([2, 5])
        assert sample[2].shape == torch.Size([2, 3, 3])
        assert len(sample[3]) == 2
        assert len(sample[3][0]) == 1
        assert isinstance(sample[3][0][0], InterventionData)

    dataset = CausalDataset(sem_sampler, 5, 7, 0)

    loader = DataLoader(dataset, batch_size=2, collate_fn=_tuple_collate_fn, drop_last=True)

    for sample in loader:
        assert sample[0]["x_0"].shape == torch.Size([2, 5, 1])
        assert sample[0].batch_size == torch.Size([2, 5])
        assert sample[1]["x_0"].shape == torch.Size([2, 5, 1])
        assert sample[1].batch_size == torch.Size([2, 5])
        assert sample[2].shape == torch.Size([2, 3, 3])
        assert sample[3] is None
