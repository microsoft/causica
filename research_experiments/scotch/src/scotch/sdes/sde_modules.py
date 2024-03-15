"""This implements the abstract classes for the drift and diffusion coefficients used in SCOTCH"""
from abc import ABC, abstractmethod

from torch import Tensor, nn


class TrajectoryGraphEncoder(ABC, nn.Module):
    @abstractmethod
    def forward(self, trajectories: Tensor, graphs: Tensor) -> Tensor:
        """Encode trajectories and graphs into context vectors for posterior SDE.

        Args:
            trajectories (Tensor): Observed trajectories, shape (batch_size, num_time_points, observed_size)
            graphs (Tensor): Weighted adjacency matrix, shape (batch_size, observed_size, observed_size)

        Returns:
            Tensor: Context vectors, shape (batch_size, num_time_points, context_size)
        """


class ContextualDriftCoefficient(ABC, nn.Module):
    @abstractmethod
    def forward(self, latent_states: Tensor, context_vectors: Tensor) -> Tensor:
        """Compute drift coefficient, given (batched) input latent states and context vectors. This will be used as
        the drift coefficient of the posterior SDE.

        Args:
            latent_states: Latent states, shape (batch_size, latent_size)
            context_vectors: Context vectors, shape (batch_size, context_size)

        Returns:
            Tensor: Drift coefficient, shape (batch_size, latent_size)
        """


class GraphCoefficient(ABC, nn.Module):
    @abstractmethod
    def forward(self, latent_states: Tensor, graphs: Tensor):
        """Compute drift (or diffusion) coefficient. The difference is that it supports the graph as input.

        Args:
            latent_states: Latent states, shape (batch_size, latent_size)
            graphs: Context vectors, shape (batch_size, latent_size, latent_size)

        Returns:
            Tensor: Drift (or diffusion) coefficient, shape (batch_size, latent_size)
        """


class DiffusionCoefficient(ABC, nn.Module):
    @abstractmethod
    def forward(self, latent_states: Tensor):
        """Compute diffusion coefficient, given (batched) input latent states.

        Args:
            latent_states: Latent states, shape (batch_size, latent_size)

        Returns:
            Tensor: Diffusion coefficient, shape (batch_size, latent_size)
        """
