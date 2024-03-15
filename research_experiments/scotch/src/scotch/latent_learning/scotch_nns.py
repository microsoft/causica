import torch
from scotch.sdes.sde_modules import (
    ContextualDriftCoefficient,
    DiffusionCoefficient,
    GraphCoefficient,
    TrajectoryGraphEncoder,
)
from torch import Tensor, nn, vmap
from torch.func import stack_module_state

from causica.functional_relationships.icgnn import FGNNI as DECIEmbedNN


class NeuralTrajectoryGraphEncoder(TrajectoryGraphEncoder):
    """Encodes trajectories and graphs into context vectors. For each time point in the trajectory, the context vector
    for that time point depends on the observed trajectory up to that point, and the graph.

    Attributes:
        gru (nn.GRU): GRU encoder for the trajectories
        lin (nn.Linear): Linear layer that maps a GRU hidden state and a flattened graph to context vectors.
        observed_size (int): Dimension of observed (X) variables in trajectory (input to encoder)
        hidden_size (int): Dimension of hidden state of GRU encoder
        context_size (int): Dimension of context vector (output of encoder)
    """

    def __init__(self, observed_size: int, hidden_size: int, context_size: int):
        """Constructor for TrajectoryEncoder."""
        super().__init__()
        self.observed_size, self.hidden_size, self.context_size = observed_size, hidden_size, context_size

        self.gru = nn.GRU(input_size=observed_size, hidden_size=hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size + observed_size * observed_size, context_size)

    def forward(self, trajectories: Tensor, graphs: Tensor) -> Tensor:
        """Encode trajectories and graphs into context vectors.

        Args:
            trajectories: Observed trajectories, shape (batch_size, num_time_points, observed_size)
            graphs: Weighted adjacency matrix, shape (batch_size, observed_size, observed_size)

        Returns:
            Tensor: Context vectors, shape (batch_size, num_time_points, context_size)
        """
        assert (
            trajectories.shape[-1] == self.observed_size
        ), "Last dimension of trajectories must match observed size of NeuralTrajectoryGraphEncoder"
        assert (
            graphs.shape[-1] == graphs.shape[-2] == self.observed_size
        ), "Last two dimensions of graphs must match observed size of NeuralTrajectoryGraphEncoder"

        num_time_points = trajectories.shape[-2]
        # shape (batch_size, num_time_points, hidden_size)
        encoded_trajectories, _ = self.gru(trajectories)
        # shape (batch_size, num_time_points, observed_size * observed_size)
        encoded_graphs = graphs.reshape(-1, 1, graphs.shape[-1] * graphs.shape[-2]).expand(-1, num_time_points, -1)

        return self.lin(torch.cat([encoded_trajectories, encoded_graphs], dim=-1))


class NeuralContextualDriftCoefficient(ContextualDriftCoefficient):
    """Implements a (family of) time-independent drift coefficient using a NN.

    The NN also takes as input a context vector, i.e. the diffusion coefficient is given by
        f(z, u): R^latent_size x R^context_size -> R^latent_size

    In SCOTCH, the context vector is used to encode the observed trajectory and graph; the drift coefficient is then
    interpreted as being the drift coefficient of the posterior SDE.

    Attributes:
        drift_fn (nn.Sequential): NN that computes drift coefficient
        latent_size (int): Dimension of latent (Z) variables
        hidden_size (int): Dimension of hidden layers of NN
        context_size (int): Dimension of context vector
    """

    def __init__(self, latent_size: int, hidden_size: int, context_size: int):
        """Constructor for NeuralDriftCoefficient."""
        super().__init__()
        self.latent_size, self.hidden_size, self.context_size = latent_size, hidden_size, context_size

        self.drift_fn = nn.Sequential(
            nn.Linear(latent_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, latent_size),
        )

    def forward(self, latent_states: Tensor, context_vectors: Tensor):
        """Compute drift coefficient, given (batched) input latent states and context vectors.

        Args:
            latent_states: Latent states, shape (batch_size, latent_size)
            context_vectors: Context vectors, shape (batch_size, context_size)

        Returns:
            Tensor: Drift coefficient, shape (batch_size, latent_size)
        """
        assert (
            latent_states.shape[-1] == self.latent_size
        ), "Last dimension of latent states must match latent size of NeuralContextualDriftCoefficient"
        assert (
            context_vectors.shape[-1] == self.context_size
        ), "Last dimension of context vectors must match context size of NeuralContextualDriftCoefficient"
        return self.drift_fn(torch.cat([latent_states, context_vectors], dim=-1))


class NeuralDiffusionCoefficient(DiffusionCoefficient):
    """Implements time-independent, diagonal (that is, elementwise) diffusion coefficient using NNs.

    g(z) = (g_1(Z_1), ..., g_d(Z_d)) : R^latent_size -> R^latent_size

    Attributes:
        diffusion_fns (nn.ModuleList): List of NNs that compute diffusion coefficient for each latent variable
        latent_size (int): Dimension of latent (Z) variables
        hidden_size (int): Dimension of hidden layers of NNs
    """

    def __init__(self, latent_size: int, hidden_size: int):
        """Constructor for NeuralDiffusionCoefficient.

        Args:
            latent_size: Dimension of latent (Z) variables
            hidden_size: Dimension of hidden layers of NNs
        """
        super().__init__()
        self.latent_size, self.hidden_size = latent_size, hidden_size

        self.diffusion_fns = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(1, hidden_size), nn.Softplus(), nn.Linear(hidden_size, 1), nn.Sigmoid()).to(
                    "cuda"
                )
                for _ in range(latent_size)
            ]
        )

        self.params, self.buffers = stack_module_state(self.diffusion_fns)

        def _diffusion_fn_wrapper(params, buffers, data):
            return torch.func.functional_call(self.diffusion_fns[0], (params, buffers), data.unsqueeze(1))

        self.diffusion_fns_vmap = vmap(
            _diffusion_fn_wrapper,
            (
                0,
                0,
                1,
            ),
            out_dims=(1,),
        )

    def forward(self, latent_states: Tensor):
        """Compute diffusion coefficient, given (batched) input latent states.

        Args:
            latent_states: Latent states, shape (batch_size, latent_size)

        Returns:
            Tensor: Diffusion coefficient, shape (batch_size, latent_size)
        """
        assert (
            latent_states.shape[-1] == self.latent_size
        ), "Last dimension of latent trajectories must match latent size of NeuralDiffusionCoefficient"

        out = self.diffusion_fns_vmap(self.params, self.buffers, latent_states).squeeze()
        return out


class DECIEmbedNNCoefficient(GraphCoefficient):
    """Defines a diffusion coefficient using DECIEmbedNN (the DECI NN model).

    The main implementational difference between this class and DECIEmbedNN is that, in the forward pass, we assume that the
    input latent_states (samples in DECIEmbedNN) and graphs have the same batch dimension; and the output also has this batch
    dimension.
    """

    def __init__(self, deci_nn: DECIEmbedNN, add_self_connections: bool = False):
        """Constructor for DECIEmbedNNDiffusionCoefficient.
        .

                Args:
                    deci_nn: DECI NN model
        """
        super().__init__()
        self.deci_nn = deci_nn
        self.f = torch.func.vmap(self.deci_nn.forward, in_dims=(0, 0), out_dims=0)
        self.add_self_connections = add_self_connections

    def forward(self, latent_states: torch.Tensor, graphs: torch.Tensor) -> torch.Tensor:
        assert latent_states.shape[:-1] == graphs.shape[:-2], "batch shapes of latent_states and graphs must match"

        if self.add_self_connections:
            final_graphs = graphs.clone()
            mask = torch.eye(final_graphs.shape[-1]).repeat(final_graphs.shape[0], 1, 1).bool()
            final_graphs[mask] = 1
        else:
            final_graphs = graphs

        return self.f(latent_states, final_graphs)
