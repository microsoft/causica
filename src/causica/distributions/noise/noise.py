import abc
from typing import Generic, Optional, TypeVar

import torch
import torch.distributions as td

from causica.distributions.distribution_module import DistributionModule

SampleType = TypeVar("SampleType")


class Noise(Generic[SampleType], abc.ABC, td.Distribution):
    """
    Extend Distributions to allow the noise (usually unparametrized) to be extracted from samples and vice versa.

    This class is used for Counterfactuals, where we want to keep the noise fixed but fix some values of the SEM.
    """

    @abc.abstractmethod
    def sample_to_noise(self, samples: SampleType) -> SampleType:
        """
        Transform from the sample observations to corresponding noise variables.

        Args:
            samples: Tensor of shape sample_shape + batch_shape + event_shape
        Returns:
            The generated samples with shape sample_shape + batch_shape + event_shape
        """

    @abc.abstractmethod
    def noise_to_sample(self, noise: SampleType) -> SampleType:
        """
        Generate samples using the given exogenous noise.

        Args:
            noise: noise variable with shape sample_shape + batch_shape.
        Returns:
            The generated samples with shape sample_shape + batch_shape + event_shape
        """


BaseNoiseType_co = TypeVar("BaseNoiseType_co", bound=Noise, covariant=True)


class IndependentNoise(Generic[BaseNoiseType_co], td.Independent, Noise[torch.Tensor]):
    """Like `td.Idenpendent` but also forwards `Noise` specific methods."""

    base_dist: BaseNoiseType_co

    def __init__(
        self,
        base_distribution: BaseNoiseType_co,
        reinterpreted_batch_ndims: int,
        validate_args: Optional[bool] = None,
    ):
        super().__init__(
            base_distribution=base_distribution,
            reinterpreted_batch_ndims=reinterpreted_batch_ndims,
            validate_args=validate_args,
        )

    def sample_to_noise(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Transform from the sample observations to corresponding noise variables.

        This just passes through to the underlying distribution.

        Args:
            samples: Tensor of shape sample_shape + batch_shape + event_shape
        Returns:
            The generated samples with shape sample_shape + batch_shape + event_shape
        """
        return self.base_dist.sample_to_noise(samples)

    def noise_to_sample(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generate samples using the given exogenous noise.

        This just passes through to the underlying distribution.

        Args:
            noise: noise variable with shape sample_shape + batch_shape.
        Returns:
            The generated samples with shape sample_shape + batch_shape + event_shape
        """
        return self.base_dist.noise_to_sample(noise)


NoiseType_co = TypeVar("NoiseType_co", bound=Noise, covariant=True)


class NoiseModule(abc.ABC, Generic[NoiseType_co], DistributionModule[NoiseType_co]):
    """Module for producing `Noise`, where sampling from the distribution corresponds to adding noise to the input.

    All subclasses must allow sampling without input, which should correspond as closely as possible to adding the noise
    to a compatible zero vector.
    """

    @abc.abstractmethod
    def forward(self, x=None) -> NoiseType_co:
        pass
