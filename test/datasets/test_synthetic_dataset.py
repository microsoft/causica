import pytest
import torch
import torch.distributions as td

from causica.data_generation.samplers.functional_relationships_sampler import LinearRelationshipsSampler
from causica.data_generation.samplers.noise_dist_sampler import (
    BernoulliNoiseModuleSampler,
    CategoricalNoiseModuleSampler,
    JointNoiseModuleSampler,
    UnivariateNormalNoiseModuleSampler,
)
from causica.data_generation.samplers.sem_sampler import SEMSampler
from causica.datasets.interventional_data import InterventionData
from causica.datasets.synthetic_dataset import CausalMetaset
from causica.distributions import ErdosRenyiDAGDistribution


@pytest.fixture(name="mixture_sem_sampler_and_shapes")
def fixture_mixture_sem_sampler_and_shapes() -> tuple[SEMSampler, dict[str, torch.Size]]:
    shapes_dict = {
        "x_0": torch.Size([1]),
        "x_1": torch.Size([5]),
        "x_2": torch.Size([1]),
        "x_3": torch.Size([3]),
        "x_4": torch.Size([4]),
    }

    noise_dist_samplers = {
        "x_0": UnivariateNormalNoiseModuleSampler(std_dist=td.Uniform(low=0.2, high=2.0), dim=1),
        "x_1": UnivariateNormalNoiseModuleSampler(std_dist=td.Uniform(low=0.2, high=2.0), dim=5),
        "x_2": BernoulliNoiseModuleSampler(base_logits_dist=td.Uniform(low=0.2, high=2.0), dim=1),
        "x_3": BernoulliNoiseModuleSampler(
            base_logits_dist=td.Uniform(low=0.2 * torch.ones([3]), high=2.0 * torch.ones([3])), dim=3
        ),
        "x_4": CategoricalNoiseModuleSampler(
            base_logits_dist=td.Uniform(low=0.2 * torch.ones([4]), high=2.0 * torch.ones([4])), num_classes=4
        ),
    }

    # Create adjacency distribution, joint noise module sampler, and functional relationships sampler
    adjacency_dist = ErdosRenyiDAGDistribution(num_nodes=5, num_edges=torch.tensor(10))
    joint_noise_module_sampler = JointNoiseModuleSampler(noise_dist_samplers)
    initial_linear_coefficient_matrix_shape = sum(shape[0] for shape in shapes_dict.values())
    functional_relationships_sampler = LinearRelationshipsSampler(
        td.Uniform(
            low=torch.ones((initial_linear_coefficient_matrix_shape, initial_linear_coefficient_matrix_shape)),
            high=3.0 * torch.ones((initial_linear_coefficient_matrix_shape, initial_linear_coefficient_matrix_shape)),
        ),
        shapes_dict,
    )

    return SEMSampler(adjacency_dist, joint_noise_module_sampler, functional_relationships_sampler), shapes_dict


def test_mixed_type_causal_metaset(mixture_sem_sampler_and_shapes: tuple[SEMSampler, dict[str, torch.Size]]):
    mixture_sem_sampler, shapes_dict = mixture_sem_sampler_and_shapes
    metaset = CausalMetaset(mixture_sem_sampler, 5, 7, 1, 2, sample_interventions=True)

    count = 0
    for sample in metaset:  # type: ignore
        count += 1
        assert sample.observations.batch_size == torch.Size([5])
        assert sample.noise.batch_size == torch.Size([5])
        for v, shape in shapes_dict.items():
            assert sample.observations[v].shape == torch.Size([5]) + shape
            assert sample.noise[v].shape == torch.Size([5]) + shape

        assert sample.graph.shape == torch.Size([5, 5])
        assert len(sample.interventions) == 1
        assert isinstance(sample.interventions[0][0], InterventionData)
        assert all((sample.observations["x_2"].unique()[:, None] == torch.tensor([0.0, 1.0])).any(dim=1))
    assert count == len(metaset)
