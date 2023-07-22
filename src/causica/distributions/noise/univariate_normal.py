from typing import Optional

import torch
import torch.distributions as td
from torch import nn

from causica.distributions.noise.noise import IndependentNoise, Noise, NoiseModule


class UnivariateNormalNoise(td.Normal, Noise[torch.Tensor]):
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


class UnivariateNormalNoiseModule(NoiseModule[IndependentNoise[UnivariateNormalNoise]]):
    """Represents a UnivariateNormalNoise with learnable parameters for independent variables."""

    def __init__(self, dim: int, init_log_scale: float = 0.0):
        """
        Args:
            dim: Number of dimensions for the NormalNoise.
        """
        super().__init__()
        self.log_scale = nn.Parameter(torch.full(torch.Size([dim]), init_log_scale))

    def forward(self, x: Optional[torch.Tensor] = None) -> IndependentNoise[UnivariateNormalNoise]:
        if x is None:
            x = torch.zeros_like(self.log_scale)
        return IndependentNoise(UnivariateNormalNoise(loc=x, scale=torch.exp(self.log_scale)), 1)
