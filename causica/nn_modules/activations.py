import torch
import torch.distributions as td
from torch import nn


class NormalDistribution(nn.Module):
    """Activation function for converting output to a normal distribution."""

    def forward(self, x: torch.Tensor) -> td.Normal:
        """Converts the input to a normal distribution using the first half of the dimensions for the mean and the
        second half for the log variance."""
        assert x.shape[-1] % 2 == 0, "Input dimension must be a multiple of 2."

        mean = x[:, : x.shape[-1] // 2]
        log_scale = x[:, x.shape[-1] // 2 :] - 4.0
        scale = torch.exp(torch.clip(log_scale, min=-20, max=5))
        return td.Normal(mean, scale)
