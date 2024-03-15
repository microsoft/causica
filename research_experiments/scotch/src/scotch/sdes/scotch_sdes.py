import torch
import torchsde
from scotch.sdes.sde_modules import ContextualDriftCoefficient, DiffusionCoefficient, GraphCoefficient
from scotch.sdes.sdes_core import SDE
from torch import Tensor, nn


def _stable_division(a, b, epsilon=1e-7):
    """This function is used to avoid division by zero."""
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


def swap_t_and_batch_dimensions(xs):
    """This function is used to swap the time and batch dimensions of a tensor. Assume the input tensor has shape
    [time, batch, ...]"""
    return xs.transpose(0, 1)


class SCOTCHPriorSDE(SDE, nn.Module):
    """Implements the prior SDE for SCOTCH (which takes graphs as input).

    The prior SDE is over latent_size variables.
        dZ_t = h(Z_t, G) dt + g(Z_t) dW_t or dZ_t = h(Z_t, G) dt + g(Z_t, G) dW_t
    where h is the prior drift coefficient and g is the diffusion coefficient.

    Requires that the graphs are specified in the constructor, such that the batch size is fixed. As such, this class
    should usually be used for transient computation, e.g. a single call of torchsde.sdeint.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, prior_drift_net: nn.Module, diffusion_net: nn.Module, graphs: Tensor):
        """Constructor for SCOTCHPriorSDE.

        Args:
            prior_drift_net: NN that computes drift coefficient of prior SDE; takes as input latent trajectories of
                shape (batch_size, latent_size) and graphs of shape (batch_size, num_nodes, num_nodes)
            diffusion_net: NN that computes diffusion coefficient of posterior SDE; takes as input latent trajectories
                of shape (batch_size, latent_size)
            graphs: Graphs used as input to prior_drift_net; shape (batch_size, num_nodes, num_nodes)
        """
        super().__init__()

        self.prior_drift_net = prior_drift_net
        self.diffusion_net = diffusion_net
        self.graphs = graphs

    def f(self, t: float, y: Tensor):
        """Computes the drift coefficient of the SDE.

        Args:
            y: Latent state of shape (batch_size, latent_size)
            t: Time point at which to evaluate drift coefficient.

        Returns:
            Drift coefficient of shape (batch_size, latent_size)
        """
        return self.prior_drift_net(y, self.graphs)

    def g(self, t: float, y: Tensor):
        """Computes the diffusion coefficient of the SDE.

        Args:
            y: Latent state of shape (batch_size, latent_size)
            t: Time point at which to evaluate diffusion coefficient.

        Returns:
            Diffusion coefficient of shape (batch_size, latent_size)
        """
        return self.diffusion_net(y, self.graphs)

    @torch.no_grad()
    def sample(self, z0: Tensor, ts: Tensor, bm: torchsde.BrownianInterval = None, dt: float = 1e-3):
        """Samples from the prior SDE.

        Args:
            z0: Initial latent state; shape (batch_size, latent_size)
            ts: Time points at which to sample; shape (num_time_points,)
            bm: Optionally, a precomputed Brownian motion process to control the randomness of the sampling
            dt: Step-size; defaults to 1e-3

        Returns:
            Latent trajectories sampled from the prior SDE; shape (batch_size, num_time_points, latent_size)
        """
        zs = torchsde.sdeint(self, z0, ts, names={"drift": "f"}, dt=dt, bm=bm)

        return swap_t_and_batch_dimensions(zs)


class AugmentedSCOTCHSDE(SDE, nn.Module):
    """Implements the augmented SDE used for training SCOTCH.

    The augmented SDE is over latent_size + 1 variables, with an auxiliary/augmented variable.
        dZ_t = f_aug(Z_t, c_t, G) dt + g_aug(Z_t, G) dW_t
    where f_aug = (f, 0.5|u|^2), g_aug = (g, 0), c_t is the context vector at time t, and:
        Posterior drift f(z, c): R^latent_size x R^context_size -> R^latent_size
        Diffusion g(z): R^latent_size -> R^latent_size
        Prior drift h(z, G): R^latent_size x R^num_nodes x R^num_nodes -> R^latent_size
        Auxiliary term u(z, c, G) = (f(z, c) - h(z, G))/g(z)

    Requires that the context vectors and graphs are specified in the constructor, such that the batch size is fixed.
    As such, this class should usually be used for transient computation, e.g. a single call of torchsde.sdeint.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        posterior_drift_net: ContextualDriftCoefficient,
        diffusion_net: DiffusionCoefficient,
        prior_drift_net: GraphCoefficient,
        ts_context_vectors: Tensor,
        graphs: Tensor,
    ):
        """Constructor for AugmentedSCOTCHSDE.

        Args:
            posterior_drift_net: NN that computes drift coefficient of posterior SDE; takes as input latent trajectories
                of shape (batch_size, latent_size) and context vectors of shape (batch_size, context_size)
            diffusion_net: NN that computes shared diffusion coefficient of prior and posterior SDE; takes as input
                latent trajectories of shape (batch_size, latent_size)
            prior_drift_net: NN that computes drift coefficient of prior SDE; takes as input latent trajectories of
                shape (batch_size, latent_size) and graphs of shape (batch_size, num_nodes, num_nodes)
            ts_context_vectors: Tuple (ts, context_vectors), where ts has shape (num_time_points,) and context_vectors
                has shape (batch_size, num_time_points, context_size) and contains the context vectors at each time.
            graphs: Graphs used as input to prior_drift_net; Tensor of shape (batch_size, num_nodes, num_nodes)
        """
        super().__init__()

        self.posterior_drift_net = posterior_drift_net
        self.diffusion_net = diffusion_net
        self.prior_drift_net = prior_drift_net

        self.ts_context_vectors = ts_context_vectors
        self.graphs = graphs

        _, context_vectors = ts_context_vectors
        assert context_vectors.shape[0] == graphs.shape[0], "Batch size of context vectors and graphs must match"

    def f(self, t: float, y: Tensor):
        """Implement drift coefficient of augmented SDE.

        Args:
            t: Time point at which to evaluate drift coefficient.
            y: Latent state of shape (batch_size, latent_size + 1).

        Returns:
            Drift coefficient of shape (batch_size, latent_size + 1).
        """
        # Remove augmented/auxiliary term
        y = y[:, :-1]

        # Retrieve context vector at first time in ts that is greater than t. As the RNN encoder (c.f.
        # TrajectoryEncoder) takes in trajectories in reverse time order, this context vector only depends on the
        # observed trajectory in the future relative to t.
        ts, context_vectors = self.ts_context_vectors
        ts_idx = min(torch.searchsorted(ts, t, right=True).item(), len(ts) - 1)
        posterior_drift = self.posterior_drift_net(y, context_vectors[:, ts_idx, :])

        # Compute diffusion and prior drift coefficients
        diffusion = self.diffusion_net(y, self.graphs)  # CHANGED with new implementatino of diffusion
        prior_drift = self.prior_drift_net(y, self.graphs)
        u = _stable_division(posterior_drift - prior_drift, diffusion)
        logqp = 0.5 * (u**2).sum(dim=1, keepdim=True)

        return torch.cat([posterior_drift, logqp], dim=-1)

    def g(self, t: float, y: Tensor):
        """Implement diffusion coefficient of augmented SDE.

        Args:
            y: Latent state of shape (batch_size, latent_size + 1).
            t: Time point at which to evaluate diffusion coefficient.

        Returns:
            Diffusion coefficient of shape (batch_size, latent_size + 1).
        """
        # Remove augmented/auxiliary term
        y = y[:, :-1]

        # Compute and return (augmented) diffusion coefficient
        diffusion = self.diffusion_net(y, self.graphs)

        return torch.nn.functional.pad(diffusion, (0, 1), value=0)
