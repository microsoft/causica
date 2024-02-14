from typing import Optional

import torch
import torch.distributions as td
from torch import nn

from causica.distributions.noise.noise import IndependentNoise, Noise, NoiseModule


class UnivariateCauchyNoise(td.Cauchy, Noise[torch.Tensor]):
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


class UnivariateCauchyNoiseModule(NoiseModule[IndependentNoise[UnivariateCauchyNoise]]):
    """Represents a UnivariateCauchyNoise with learnable parameters for independent variables."""

    def __init__(self, dim: int, init_log_scale: float | torch.Tensor = 0.0):
        """
        Args:
            dim: Number of dimensions for the NormalNoise.
        """
        super().__init__()
        if isinstance(init_log_scale, torch.Tensor):
            if init_log_scale.squeeze().ndim == 0:
                init_log_scale = torch.full(torch.Size([dim]), init_log_scale.item())
            else:
                assert init_log_scale.ndim == 1
                assert init_log_scale.shape[0] == dim
        else:
            init_log_scale = torch.full(torch.Size([dim]), init_log_scale)

        self.log_scale = nn.Parameter(init_log_scale)

    def forward(self, x: Optional[torch.Tensor] = None) -> IndependentNoise[UnivariateCauchyNoise]:
        if x is None:
            x = torch.zeros_like(self.log_scale)
        return IndependentNoise(UnivariateCauchyNoise(loc=x, scale=torch.exp(self.log_scale)), 1)
