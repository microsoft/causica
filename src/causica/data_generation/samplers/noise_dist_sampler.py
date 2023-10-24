import abc
from typing import Mapping

import torch
import torch.distributions as td

from causica.data_generation.samplers.sampler import Sampler
from causica.distributions import JointNoiseModule
from causica.distributions.noise import NoiseModule, UnivariateNormalNoiseModule
from causica.distributions.noise.bernoulli import BernoulliNoiseModule


class NoiseModuleSampler(Sampler[NoiseModule]):
    """
    An interface of a univariate noise sampler
    """

    @abc.abstractmethod
    def sample(
        self,
    ) -> NoiseModule:
        """Sample a sample type with given shape"""


class JointNoiseModuleSampler(NoiseModuleSampler):
    """Sampler for JointNoiseModule, given shapes and types of different variables"""

    def __init__(
        self,
        noise_dist_samplers: Mapping[str, NoiseModuleSampler],
    ):
        super().__init__()
        self.noise_dist_samplers = noise_dist_samplers

    def sample(self) -> JointNoiseModule:
        noise_modules = {}
        for key, noise_sampler in self.noise_dist_samplers.items():
            noise_modules[key] = noise_sampler.sample()
        return JointNoiseModule(independent_noise_modules=noise_modules)


class UnivariateNormalNoiseModuleSampler(NoiseModuleSampler):
    """Sample a UnivariateNormalNoiseModule, with standard deviation given by a distribution."""

    def __init__(self, std_dist: td.Distribution, dim: int = 1):
        super().__init__()
        self.std_dist = std_dist
        self.dim = dim

    def sample(
        self,
    ):
        return UnivariateNormalNoiseModule(dim=self.dim, init_log_scale=torch.log(self.std_dist.sample()).item())


class BernoulliNoiseModuleSampler(NoiseModuleSampler):
    """Sample a BernoulliNoiseModule, with base_logits given by a distribution."""

    def __init__(self, base_logits_dist: td.Distribution, dim: int = 1):
        super().__init__()
        self.base_logits_dist = base_logits_dist
        self.dim = dim

    def sample(
        self,
    ) -> NoiseModule:
        base_logits = self.base_logits_dist.sample().item()
        return BernoulliNoiseModule(dim=self.dim, init_base_logits=base_logits)
