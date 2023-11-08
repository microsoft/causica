from typing import Optional

import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import nn
from torch.distributions import OneHotCategorical

from causica.distributions.noise.noise import Noise, NoiseModule


class CategoricalNoise(OneHotCategorical, Noise):
    def __init__(self, delta_logits: torch.Tensor, base_logits: torch.Tensor):
        """
        A Categorical distribution with parameters defined by base_logits and self.delta_logits (predictions from an NN).

        Args:
            delta_logits: Tensor with shape [sample_shape, event_shape]
            base_logits: Tensor with shape [event_shape] where event_shape shows the number of categories for this node
        """
        self.delta_logits = delta_logits
        super().__init__(logits=base_logits + delta_logits, validate_args=False)

    def sample_to_noise(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Transform from the sample observations to corresponding noise variables.

        This will draw from the noise posterior given the observations

        A posterior sample of the Gumbel noise random variables given observation x and probabilities
        `self.base_logits + logit_deltas`.

        This methodology is described in https://arxiv.org/pdf/1905.05824.pdf.
        See https://cmaddis.github.io/gumbel-machinery for derivation of Gumbel posteriors.
        For a derivation of this exact algorithm using softplus, see https://www.overleaf.com/8628339373sxjmtvyxcqnx.

        Args:
            samples: Tensor of shape sample_shape + batch_shape + event_shape
        Returns:
            The generated samples with shape sample_shape + batch_shape + event_shape
        """

        device = self.delta_logits.device
        dist = td.Gumbel(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        top_sample = dist.sample(samples.shape[:-1] + (1,)) + self.logits.logsumexp(-1, keepdim=True)
        lower_samples = dist.sample(samples.shape) + self.logits
        lower_samples[samples == 1] = float("inf")
        return top_sample - F.softplus(top_sample - lower_samples) - self.delta_logits

    def noise_to_sample(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generate samples using the given exogenous noise.

        Args:
            noise: noise variable with shape sample_shape + batch_shape.
        Returns:
            The generated samples with shape sample_shape + batch_shape + event_shape
        """
        x = self.delta_logits + noise
        maxes = torch.max(x, dim=-1, keepdim=True)
        return (x >= maxes[0]).float()


class CategoricalNoiseModule(NoiseModule[CategoricalNoise]):
    """Represents a CategoricalNoise distribution with learnable logits."""

    def __init__(self, num_classes: int, init_base_logits: torch.Tensor | None = None):
        """
        Args:
            num_classes: Number of classes.
            init_base_logits: Initial base logits.
        """
        super().__init__()

        if init_base_logits is not None:
            assert init_base_logits.ndim == 1
            assert init_base_logits.shape[0] == num_classes
        else:
            init_base_logits = torch.zeros(num_classes)

        self.base_logits = nn.Parameter(init_base_logits)

    def forward(self, x: Optional[torch.Tensor] = None) -> CategoricalNoise:
        if x is None:
            x = torch.zeros_like(self.base_logits)
        return CategoricalNoise(delta_logits=x, base_logits=self.base_logits)
