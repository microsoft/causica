import abc

import torch
import torch.distributions as td


class NoiseAccessible(abc.ABC):
    """
    Extend Distributions to allow the noise (usually unparametrized) to be extracted from samples and vice versa.

    This class is used for Counterfactuals, where we want to keep the noise fixed but fix some values of the SEM.
    """

    @abc.abstractmethod
    def sample_to_noise(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Transform from the sample observations to corresponding noise variables.

        Args:
            samples: Tensor of shape sample_shape + batch_shape + event_shape
        Returns:
            The generated samples with shape sample_shape + batch_shape + event_shape
        """
        pass

    @abc.abstractmethod
    def noise_to_sample(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generate samples using the given exogenous noise.

        Args:
            noise: noise variable with shape sample_shape + batch_shape.
        Returns:
            The generated samples with shape sample_shape + batch_shape + event_shape
        """
        pass


class NoiseAccessibleDistribution(NoiseAccessible, td.Distribution):
    """
    A class used for type hinting.

    Do not inherit from this class. Instead, inherit from the distribution you're interested in
    and mixin the noise accessible class.
    """


class NoiseAccessibleIndependent(td.Independent, NoiseAccessible):
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
