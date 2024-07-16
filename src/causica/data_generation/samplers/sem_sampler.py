from typing import Optional

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
        log_functional_rescaling_sampler: Optional[FunctionalRelationshipsSampler] = None,
    ):
        self.adjacency_dist = adjacency_dist
        self.joint_noise_module_sampler = joint_noise_module_sampler
        self.functional_relationships_sampler = functional_relationships_sampler
        self.log_functional_rescaling_sampler = log_functional_rescaling_sampler
        self.shapes_dict: dict[str, torch.Size] = functional_relationships_sampler.shapes_dict

    def sample(self):
        adjacency_matrix = self.adjacency_dist.sample()
        functional_relationships = self.functional_relationships_sampler.sample()
        log_func_rescale = (
            self.log_functional_rescaling_sampler.sample()
            if self.log_functional_rescaling_sampler is not None
            else None
        )
        joint_noise_module = self.joint_noise_module_sampler.sample()
        return DistributionParametersSEM(
            graph=adjacency_matrix,
            noise_dist=joint_noise_module,
            func=functional_relationships,
            log_func_rescale=log_func_rescale,
        )
