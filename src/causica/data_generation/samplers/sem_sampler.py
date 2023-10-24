import torch

from causica.data_generation.samplers.functional_relationships_sampler import FunctionalRelationshipsSampler
from causica.data_generation.samplers.noise_dist_sampler import JointNoiseModuleSampler
from causica.data_generation.samplers.sampler import Sampler
from causica.distributions import AdjacencyDistribution
from causica.sem.distribution_parameters_sem import DistributionParametersSEM


class SEMSampler(Sampler[DistributionParametersSEM]):
    """Sample a SEM given adjacency, a JointNoiseModuleSampler and functional relationships distributions."""

    def __init__(
        self,
        adjacency_dist: AdjacencyDistribution,
        joint_noise_module_sampler: JointNoiseModuleSampler,
        functional_relationships_sampler: FunctionalRelationshipsSampler,
    ):
        self.adjacency_dist = adjacency_dist
        self.joint_noise_module_sampler = joint_noise_module_sampler
        self.functional_relationships_sampler = functional_relationships_sampler
        self.shapes_dict: dict[str, torch.Size] = functional_relationships_sampler.shapes_dict

    def sample(self):
        adjacency_matrix = self.adjacency_dist.sample()
        functional_relationships = self.functional_relationships_sampler.sample()
        joint_noise_module = self.joint_noise_module_sampler.sample()
        return DistributionParametersSEM(
            graph=adjacency_matrix, noise_dist=joint_noise_module, func=functional_relationships
        )
