"""
Module that provides data normalisation functionality.
"""
import torch
import torch.distributions as td
from tensordict import TensorDict
from torch import nn

from causica.distributions.transforms import JointTransform


class SingleVariableStandardizer(nn.Module):
    """Standardizer module for a single variable, ie a single tensor."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Args:
            mean: Mean of the variable
            std: Standard deviation of the variable
        """
        super().__init__()

        if mean.shape != std.shape:
            raise ValueError("mean and std must have the same shape.")

        self.mean: torch.Tensor
        self.std: torch.Tensor

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def __call__(self) -> td.AffineTransform:
        return super().__call__()

    def forward(self) -> td.AffineTransform:
        return td.AffineTransform(loc=-self.mean / self.std, scale=1 / self.std)


class JointStandardizer(nn.Module):
    """Standardizer module for TensorDicts."""

    def __init__(self, means: TensorDict, stds: TensorDict):
        """
        Args:
            means: Means of the variables
            stds: Standard deviations of the variables
        """
        super().__init__()

        if set(means.keys()) != set(stds.keys()):
            raise ValueError("Requires means and stds to have the same keys.")

        if not all(m.shape == s.shape for m, s in zip(means.values(), stds.values())):
            raise ValueError("Requires means and stds to have the same shapes.")

        self.transform_modules = nn.ModuleDict(
            {key: SingleVariableStandardizer(means[key], stds[key]) for key in means.keys()}
        )

    def forward(self) -> JointTransform:
        return JointTransform({key: module() for key, module in self.transform_modules.items()})


def fit_standardizer(data: TensorDict) -> JointStandardizer:
    """Calculate the mean and standard deviation over the first dimension of each variable in the TensorDict and return a standardizer."""
    means = data.apply(lambda x: torch.mean(x, dim=0, keepdim=True), batch_size=(1,))
    # Filter out std == 0
    stds = data.apply(lambda x: torch.std(x, dim=0, keepdim=True), batch_size=(1,)).apply(
        lambda x: torch.where(x == 0, torch.ones_like(x), x)
    )
    return JointStandardizer(means=means, stds=stds)
