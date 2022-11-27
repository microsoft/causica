from typing import List, Optional, Tuple

import pyro.distributions as distrib
import torch
import torch.distributions as td
from pyro.distributions.conditional import ConditionalTransform
from pyro.distributions.transforms import ComposeTransform
from pyro.distributions.transforms.spline import ConditionalSpline, Spline
from torch import nn

from .diagonal_flows import AffineDiagonalPyro, create_diagonal_spline_flow
from .generation_functions import TemporalHyperNet


# TODO: Add tests for base distributions, and ensure we are not duplicating any pytorch.distributions functionality unnecessarily
class GaussianBase(nn.Module):
    def __init__(self, input_dim: int, device: torch.device, train_base: bool = True, log_scale_init: float = 0.0):
        """
        Gaussian base distribution with 0 mean and optionally learnt variance. The distribution is factorised to ensure SEM invertibility.
            The mean is fixed. This class provides an interface analogous to torch.distributions, exposing .sample and .log_prob methods.

        Args:
            input_dim: dimensionality of observations
            device: torch.device on which object should be stored
            train_base: whether to fix the variance to 1 or learn it with gradients
            log_scale_init: initial value for Gaussian log scale values
        """
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.log_scale_init = log_scale_init
        self.mean_base, self.logscale_base = self._initialize_params_base_dist(train=train_base)

    def _initialize_params_base_dist(self, train: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the parameters of the base distribution.

        Args:
            train: Whether the distribution's parameters are trainable or not.
        """

        mean = nn.Parameter(torch.zeros(self.input_dim, device=self.device), requires_grad=False)
        logscale = nn.Parameter(
            self.log_scale_init * torch.ones(self.input_dim, device=self.device), requires_grad=train
        )
        return mean, logscale

    def log_prob(self, z: torch.Tensor):
        """
        Returns a the log-density per sample and dimension of z

        Args:
            z (batch, input_dim)

        Returns:
            log_prob (batch, input_dim)
        """
        dist = td.Normal(self.mean_base, torch.exp(self.logscale_base))
        return dist.log_prob(z)

    def sample(self, Nsamples: int):
        """
        Draw samples

        Args:
            Nsamples

        Returns:
            samples (Nsamples, input_dim)
        """
        dist = td.Normal(self.mean_base, torch.exp(self.logscale_base))
        return dist.sample((Nsamples,))


class DiagonalFlowBase(nn.Module):
    def __init__(self, input_dim: int, device: torch.device, num_bins: int = 8, flow_steps: int = 1) -> None:
        """
        Learnable base distribution based on a composite affine-spline transformation of a standard Gaussian. The distribution is factorised to ensure SEM invertibility.
           This means that the flow acts dimension-wise, without sharing information across dimensions.
           This class provides an interface analogous to torch.distributions, exposing .sample and .log_prob methods.

        Args:
            input_dim: dimensionality of observations
            device: torch.device on which object should be stored
            num_bins: ow many bins to use for spline transformation
            flow_steps: how many affine-spline steps to take. Recommended value is 1.
        """
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.transform = create_diagonal_spline_flow(
            flow_steps=flow_steps, features=self.input_dim, num_bins=num_bins, tail_bound=3
        ).to(self.device)
        self.base_dist = td.Normal(
            loc=torch.zeros(self.input_dim, device=self.device),
            scale=torch.ones(self.input_dim, device=self.device),
            validate_args=None,
        )

    def log_prob(self, z: torch.Tensor):
        """
        Returns a the log-density per sample and dimension of z

        Args:
            z (batch, input_dim)

        Returns:
            log_prob (batch, input_dim)
        """
        u, logdet = self.transform.inverse(z)
        log_pu = self.base_dist.log_prob(u)
        return logdet + log_pu

    def sample(self, Nsamples: int):
        """
        Draw samples

        Args:
            Nsamples

        Returns:
            samples (Nsamples, input_dim)
        """
        with torch.no_grad():
            u = self.base_dist.sample((Nsamples,))
            z, _ = self.transform.forward(u)
        return z


class TemporalConditionalSplineFlow(nn.Module):
    """
    This implements the conditional spline flow transformed distribution, where the flow parameters are
    predicted with a temporal hypernet and lagged parents value.
    """

    def __init__(
        self,
        cts_node: List[int],
        group_mask: torch.Tensor,
        device: torch.device,
        lag: int,
        num_bins: int = 8,
        additional_flow: int = 0,
        norm_layers: bool = True,
        res_connection: bool = True,
        embedding_size: Optional[int] = None,
        out_dim_g: Optional[int] = None,
        layers_g: Optional[List[int]] = None,
        layers_f: Optional[List[int]] = None,
        order: str = "quadratic",
    ) -> None:
        """
        This initializes the conditional spline flow based transformed distribution with a hyper net. The transformation
        consists of an affine transformation followed by the conditional spline flow. Optionally, one can specify additional
        transformations consisting of affine + spline flows. In the end, we append an affine transform as the last layer.
        Args:
            input_dim: the dimension of conditional spline flow.
            cts_node: A list of node idx specifies the cts variables.
            group_mask: the group mask of the variables
            device: Device to use
            lag: the lag of the model, should be consistent with ar-deci.
            num_bins: num of bins for spline flow
            additional_flow: number of additional Spline flow (unconditioned) on top of the conditioned spline flow.
                0 means no additional flow.
            norm_layers: whether to use normalization layer in hypernet.
            res_connection: whether to use re_connection in hypernet.
            embedding_size: the size of node embedding in hypernet
            out_dim_g: the output dimension of g in hypernet.
            layers_f: decoder layer sizes in hypernet.
            layers_g: encoder layer sizes in hypernet.
            order: the transformation order of spline flow.
        """
        super().__init__()
        self.device = device
        self.num_bins = num_bins
        # Initialize a temporal hypernetwork for parameters prediction of conditional spline
        self.cts_node = cts_node
        self.cts_dim = len(cts_node)
        self.lag = lag
        self.norm_layer = nn.LayerNorm if norm_layers else None
        self.res_connection = res_connection
        self.order = order
        if self.order == "quadratic":
            param_dim = [
                self.num_bins,
                self.num_bins,
                (self.num_bins - 1),
            ]  # this is for quadratic order conditional spline flow
        elif self.order == "linear":
            param_dim = [
                self.num_bins,
                self.num_bins,
                (self.num_bins - 1),
                self.num_bins,
            ]  # this is for linear order conditional spline flow
        else:
            raise ValueError("The order of spline flow can either be 'linear' or 'quadratic' ")
        self.hypernet = TemporalHyperNet(
            cts_node=self.cts_node,
            group_mask=group_mask,
            device=self.device,
            lag=self.lag,
            param_dim=param_dim,
            norm_layer=self.norm_layer,
            res_connection=self.res_connection,
            embedding_size=embedding_size,
            out_dim_g=out_dim_g,
            layers_f=layers_f,
            layers_g=layers_g,
        ).to(device)

        self.additional_flow = additional_flow
        # The default base dist is Gaussian with zero mean and unit variance
        self.base_dist = distrib.Normal(
            torch.zeros(self.cts_dim, device=device), torch.ones(self.cts_dim, device=device)
        )
        # Placeholder for conditional spline flow
        self.transform = nn.ModuleList(
            [
                AffineDiagonalPyro(input_dim=self.cts_dim).to(device),
                ConditionalSpline(
                    self.hypernet, input_dim=self.cts_dim, count_bins=self.num_bins, order=self.order, bound=5.0
                ).to(device),
            ]
        )

        # Additional Flow
        if additional_flow > 0:
            for _ in range(additional_flow):
                self.transform.append(AffineDiagonalPyro(input_dim=self.cts_dim).to(device))
                self.transform.append(
                    Spline(input_dim=self.cts_dim, count_bins=self.num_bins, order="quadratic", bound=5.0).to(device)
                )
        # Diagonal affine transform as the last layer.
        self.transform.append(AffineDiagonalPyro(input_dim=self.cts_dim).to(device))

    def log_prob(self, X_input: torch.Tensor, X_history: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        This computes the conditional log likelihood of X_input based on the conditional spline flow transformed
        distribution given X_history and weighted adjacency W.
        Args:
            X_input: Tensor with shape [batch, input_dim]
            X_history: History observations with shape [batch, model_lag, proc_dim_all], note proc_dim_all >= input_dim
                due to binary and categorical variables.
            W: Weighted adjacency matrix with shape [lag+1, num_node, num_node]
        Returns:
            log_likelihood: Tensor with shape [batch, input_dim]
        """
        # Transform conditional placeholder to actual conditional distribution
        context_dict = {"X": X_history, "W": W}
        flow_dist = distrib.ConditionalTransformedDistribution(self.base_dist, self.transform).condition(context_dict)
        return flow_dist.log_prob(X_input)

    def sample(self, Nsamples: int, X_history: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        This is to draw noise samples from the conditional spline flow based on lagged X_history and adjacency W.
        Args:
            Nsamples: The number of noise samples for each batch X_history
            X_history: Lagged observation with shape [batch, lag, proc_dim_all], note proc_dim_all >= input_dim
                due to binary and categorical variables.
            W: The weighted adjacency matrix with shape [lag+1, num_node, num_node]
        Returns:
            noise samples: Tensor with shape [Nsamples, batch, input_dim]
        """
        # Transform conditional placeholder to actual conditional distribution
        context_dict = {"X": X_history, "W": W}
        batch = X_history.shape[0]
        flow_dist = distrib.ConditionalTransformedDistribution(self.base_dist, self.transform).condition(context_dict)
        return flow_dist.sample([Nsamples, batch])

    def transform_noise(self, Z: torch.Tensor, X_history: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """
        This will transforms the noise from base distribution to the noise from the flow distribution. This is achieved
        by generating the torch conditional distribution, then pass the Z through the transforms.
        Args:
            Z: [Nsamples, batch, input_dim] or [batch, input_dim], where input_dim == len(cts_node). Nsamples is the number of noise sample
                per batch obs. The batch must match the batch size in X_history.
            X_history: Lagged observation with shape [batch, lag, proc_dim_all], note proc_dim_all >= input_dim
                due to binary and categorical variables.
            W: The weighted adjacency matrix with shape [lag+1, num_node, num_node]

        Returns:
            transformed_noise: shape [Nsamples, batch, input_dim] or [batch, input_dim]
        """
        context_dict = {"X": X_history, "W": W}
        conditional_transform = [
            t.condition(context_dict) if isinstance(t, ConditionalTransform) else t for t in self.transform
        ]
        transform = ComposeTransform(conditional_transform)
        Z = transform(Z)
        return Z


class CategoricalLikelihood(nn.Module):
    def __init__(self, input_dim: int, device: torch.device):
        """
        Discrete likelihood model. This model learns a base probability distribution.
        At evaluation time, it takes in an additional input which is added to this base distribution.
        This allows us to model both conditional and unconditional nodes.

        Args:
            input_dim: dimensionality of observations
            device: torch.device on which object should be stored
        """
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.base_logits = nn.Parameter(torch.zeros(self.input_dim, device=self.device), requires_grad=True)
        self.softplus = nn.Softplus()

    def log_prob(self, x: torch.Tensor, logit_deltas: torch.Tensor):
        """
        The modelled probability distribution for x is given by `self.base_logits + logit_deltas`.
        This method returns the log-density per sample.

        Args:
            x (batch, input_dim): Should be one-hot encoded
            logit_deltas (batch, input_dim)

        Returns:
            log_prob (batch, input_dim)
        """
        dist = td.OneHotCategorical(logits=self.base_logits + logit_deltas, validate_args=False)
        return dist.log_prob(x)

    def sample(self, n_samples: int):
        """
        Samples Gumbels that can be used to sample this variable using the Gumbel max trick.
        This method does **not** return hard-thresholded samples.
        Args:
            n_samples

        Returns:
            samples (Nsamples, input_dim)
        """
        dist = td.Gumbel(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        return dist.sample((n_samples, self.input_dim)) + self.base_logits

    def posterior(self, x: torch.Tensor, logit_deltas: torch.Tensor):
        """
        A posterior sample of the Gumbel noise random variables given observation x and probabilities
        `self.base_logits + logit_deltas`.
        This methodology is described in https://arxiv.org/pdf/1905.05824.pdf.
        See https://cmaddis.github.io/gumbel-machinery for derivation of Gumbel posteriors.
        For a derivation of this exact algorithm using softplus, see https://www.overleaf.com/8628339373sxjmtvyxcqnx.

        Args:
            x (batch, input_dim): Should be one-hot encoded
            logit_deltas (batch, input_dim)

        Returns:
            z (batch, input_dim)
        """
        logits = self.base_logits + logit_deltas
        dist = td.Gumbel(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        top_sample = dist.sample((x.shape[0], 1)) + logits.logsumexp(-1, keepdim=True)
        lower_samples = dist.sample((x.shape[0], self.input_dim)) + logits
        lower_samples[x == 1] = float("inf")
        samples = top_sample - self.softplus(top_sample - lower_samples) - logits
        return samples + self.base_logits


class BinaryLikelihood(nn.Module):
    def __init__(self, input_dim: int, device: torch.device):
        """
        Binary likelihood model. This model learns a base probability distribution.
        At evaluation time, it takes in an additional input which is added to this base distribution.
        This allows us to model both conditional and unconditional nodes.

        Args:
            input_dim: dimensionality of observations
            device: torch.device on which object should be stored
        """
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.base_logits = nn.Parameter(torch.zeros(self.input_dim, device=self.device), requires_grad=True)
        self.softplus = nn.Softplus()

    def log_prob(self, x: torch.Tensor, logit_deltas: torch.Tensor):
        """
        The modelled probability distribution for x is given by `self.base_logits + logit_deltas`.
        This method returns the log-density per sample.

        Args:
            x (batch, input_dim): Should be one-hot encoded
            logit_deltas (batch, input_dim)

        Returns:
            log_prob (batch, input_dim)
        """
        dist = td.Bernoulli(logits=self.base_logits + logit_deltas, validate_args=False)
        return dist.log_prob(x)

    def sample(self, n_samples: int):
        """
        Samples a Logistic random variable that can be used to sample this variable.
        This method does **not** return hard-thresholded samples.
        Args:
            n_samples

        Returns:
            samples (Nsamples, input_dim)
        """
        # The difference of two independent Gumbel(0, 1) variables is a Logistic random variable
        dist = td.Gumbel(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        g0 = dist.sample((n_samples, self.input_dim))
        g1 = dist.sample((n_samples, self.input_dim))
        return g1 - g0 + self.base_logits

    def posterior(self, x: torch.Tensor, logit_deltas: torch.Tensor):
        """
        A posterior sample of the logistic noise random variables given observation x and probabilities
        `self.base_logits + logit_deltas`.

        Args:
            x (batch, input_dim): Should be one-hot encoded
            logit_deltas (batch, input_dim)

        Returns:
            z (batch, input_dim)
        """
        logits = self.base_logits + logit_deltas
        dist = td.Gumbel(torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
        top_sample = dist.sample(x.shape)
        neg_log_prob_non_sampled = self.softplus(logits * x - logits * (1 - x))
        positive_sample = self.softplus(top_sample - dist.sample(x.shape) + neg_log_prob_non_sampled)
        sample = positive_sample * x - positive_sample * (1 - x) - logits
        return sample + self.base_logits
