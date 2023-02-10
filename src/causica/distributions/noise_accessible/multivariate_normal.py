import torch
import torch.distributions as td

from causica.distributions.noise_accessible.noise_accessible import NoiseAccessibleDistribution


class NoiseAccessibleMultivariateNormal(td.MultivariateNormal, NoiseAccessibleDistribution):
    def sample_to_noise(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Transform from the sample observations to corresponding noise variables.

        Args:
            samples: Tensor of shape sample_shape + batch_shape + event_shape
        Returns:
            The generated samples with shape sample_shape + batch_shape + event_shape
        """
        return samples - self.loc

    def noise_to_sample(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generate samples using the given exogenous noise.

        Args:
            noise: noise variable with shape sample_shape + batch_shape.
        Returns:
            The generated samples with shape sample_shape + batch_shape + event_shape
        """
        return noise + self.loc
