"""This implements the abstract template for stochastic differential equations"""
from abc import ABC, abstractmethod

import torch


class SDE(ABC):
    """Abstract base class for stochastic differential equations.

    Any concrete class must implement the f and g methods, and noise_type and sde_type properties.
    """

    @property
    @abstractmethod
    def noise_type(self) -> str:
        """Type of SDE noise.

        Must be one of "diagonal", "additive", "scalar". or "general".
        """

    @property
    @abstractmethod
    def sde_type(self):
        """Type of SDE.

        Must be one of "ito" or "stratonovich".
        """

    @abstractmethod
    def f(self, t: float, y: torch.Tensor) -> torch.Tensor:
        """Drift coefficient of the SDE.

        Args:
            t: time point; scalar
            y: variable values; tensor of size (batch_size, state_size)
        """

    @abstractmethod
    def g(self, t: float, y: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient of the SDE.

        Args:
            t: time point; scalar
            y: variable values; tensor of size (batch_size, state_size)
        """
