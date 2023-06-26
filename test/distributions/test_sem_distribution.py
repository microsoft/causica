import pytest
import torch

from causica.distributions.adjacency.adjacency_distributions import AdjacencyDistribution
from causica.distributions.adjacency.enco import ENCOAdjacencyDistribution
from causica.distributions.noise.joint import JointNoiseModule
from causica.distributions.noise.noise import Noise, NoiseModule
from causica.distributions.noise.univariate_normal import UnivariateNormalNoiseModule
from causica.functional_relationships import ICGNN
from causica.functional_relationships.functional_relationships import FunctionalRelationships
from causica.sem.sem_distribution import SEMDistribution


def create_sem_params(
    shapes: dict[str, torch.Size]
) -> tuple[AdjacencyDistribution, JointNoiseModule, FunctionalRelationships]:
    num_nodes = len(shapes)
    independent_noise_modules: dict[str, NoiseModule[Noise[torch.Tensor]]] = {
        name: UnivariateNormalNoiseModule(shape[-1]) for name, shape in shapes.items()
    }
    noise_dist = JointNoiseModule(independent_noise_modules)
    func = ICGNN(shapes)
    logits_exist = torch.randn((num_nodes, num_nodes))
    logits_orient = torch.randn(((num_nodes * (num_nodes - 1)) // 2,))
    adjacency_dist = ENCOAdjacencyDistribution(logits_exist=logits_exist, logits_orient=logits_orient)
    return adjacency_dist, noise_dist, func


@pytest.mark.parametrize("sample_shape", [torch.Size([]), torch.Size([2]), torch.Size([5, 4])])
def test_sem_distribution_samples(sample_shape: torch.Size):
    """Test sampling."""
    shapes = {"a": torch.Size([5]), "b": torch.Size([2])}
    sem_dist = SEMDistribution(*create_sem_params(shapes))
    assert len(sem_dist.sample(sample_shape)) == sample_shape.numel()
    assert len(sem_dist.relaxed_sample(sample_shape, temperature=0.5)) == sample_shape.numel()


def test_sem_distribution_passthrough():
    """Test distribution properties."""
    shapes = {"a": torch.Size([5]), "b": torch.Size([2])}
    adjacency_dist, noise_dist, func = create_sem_params(shapes)
    sem_dist = SEMDistribution(adjacency_dist, noise_dist, func)
    torch.testing.assert_close(adjacency_dist.entropy(), sem_dist.entropy())
    torch.testing.assert_close(adjacency_dist.mean, sem_dist.mean.graph)
    torch.testing.assert_close(adjacency_dist.mode, sem_dist.mode.graph)
    torch.testing.assert_close(adjacency_dist.log_prob(adjacency_dist.mode), sem_dist.log_prob(sem_dist.mode))
