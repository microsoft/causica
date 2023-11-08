from typing import Optional

import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import nn

from causica.distributions.noise.noise import IndependentNoise, Noise, NoiseModule


class BernoulliNoise(td.Bernoulli, Noise):
    def __init__(self, delta_logits: torch.Tensor, base_logits: torch.Tensor):
        """
        A Bernoulli distribution with parameters defined by base_logits and x_hat (predictions for noiseless value).

        Args:
            delta_logits: Tensor with shape sample_shape + batch_shape. These are the predicted values.
            base_logits: Tensor with shape batch_shape
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
        assert (
            samples.shape == self.delta_logits.shape
        ), "The shape of the input does not match the shape of the logit_deltas"
        device = self.delta_logits.device
        dist = td.Gumbel(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
        diff_sample = dist.sample(samples.shape) - dist.sample(samples.shape)  # sample_shape + batch_shape
        neg_log_prob_non_sampled = F.softplus(self.logits * samples - self.logits * (1 - samples))
        positive_sample = F.softplus(diff_sample + neg_log_prob_non_sampled)
        return positive_sample * samples - positive_sample * (1 - samples) - self.delta_logits

    def noise_to_sample(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generate samples using the given exogenous noise.

        Args:
            noise: noise variable with shape sample_shape + batch_shape.
        Returns:
            The generated samples with shape sample_shape + batch_shape + event_shape
        """
        return ((self.delta_logits + noise) > 0).float()

    @property
    def mode(self):
        """
        Override the default `mode` method to prevent it returning nan's.

        We favour sparseness, so if logit == 0, set the mode to be zero.
        """
        return (self.logits > 0).to(self.logits, non_blocking=True)


class BernoulliNoiseModule(NoiseModule[IndependentNoise[BernoulliNoise]]):
    """Represents a BernoulliNoise distribution with learnable logits."""

    def __init__(self, dim: int, init_base_logits: float | torch.Tensor = 0.0):
        """
        Args:
            dim: Number of dimensions (independent Bernouilli's).
        """
        super().__init__()
        if isinstance(init_base_logits, torch.Tensor):
            if init_base_logits.squeeze().ndim == 0:
                init_base_logits = torch.full(torch.Size([dim]), init_base_logits.item())
            else:
                assert init_base_logits.ndim == 1
                assert init_base_logits.shape[0] == dim
        else:
            init_base_logits = torch.full(torch.Size([dim]), init_base_logits)

        self.base_logits = nn.Parameter(init_base_logits)

    def forward(self, x: Optional[torch.Tensor] = None) -> IndependentNoise[BernoulliNoise]:
        if x is None:
            x = torch.zeros_like(self.base_logits)
        return IndependentNoise(BernoulliNoise(delta_logits=x, base_logits=self.base_logits), 1)
