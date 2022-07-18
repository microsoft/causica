"""Classes to enable models to optimise treatments."""
from typing import Union

import torch


class IdentityTreatmentPolicy(torch.nn.Module):
    """Create a treatment from a tensor."""

    def __init__(self, treatment: torch.Tensor, learnable: bool = True) -> None:
        super().__init__()
        if learnable:
            self.register_parameter("treatment", torch.nn.Parameter(treatment))
        else:
            self.register_buffer("treatment", treatment)

    def forward(self) -> Union[torch.Tensor, torch.nn.Module]:
        """Return the parameter unmodified"""
        return self.treatment


class GumbelSoftMaxPolicy(torch.nn.Module):
    """Parameterise a one-hot encoded treatment as a Gumbel softmax"""

    def __init__(self, log_prob: torch.Tensor, hard: bool = True, tau: float = 0.1, learnable: bool = True) -> None:
        super().__init__()
        if learnable:
            self.register_parameter("log_prob", torch.nn.Parameter(log_prob))
        else:
            self.register_buffer("log_prob", log_prob)
        self.hard = hard
        self.tau = tau

    @property
    def treatment(self) -> torch.Tensor:
        return self.forward().detach()

    def forward(self) -> torch.Tensor:
        return torch.nn.functional.gumbel_softmax(self.log_prob, tau=self.tau, hard=self.hard)  # type: ignore


class RandomTreatmentPolicy(torch.nn.Module):
    """Random treatment policy"""

    def __init__(self, treatment_distribution: torch.distributions.Distribution) -> None:
        """treatment_distribution: distribution to sample treatments from"""
        super().__init__()
        self.treatment_distribution = treatment_distribution

    def forward(self) -> torch.Tensor:
        return self.treatment_distribution.sample()
