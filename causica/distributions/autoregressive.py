from typing import Optional, Tuple, Type

import numpy as np
import torch
import torch.distributions as td

from causica.distributions.utils import fill_triangular, unfill_triangular

from .adjacency_distributions import AdjacencyDistribution


class LSTM(torch.nn.Module):
    """
    An LSTM implementation for autoregressive modeling of the adjacency matrix, as described in https://arxiv.org/abs/2106.07635.
    """

    def __init__(self, num_nodes: int, **kwargs):
        """
        Args:
                num_nodes: The number of nodes of the graph to parameterize.
                hidden_dim: The hidden dimension of the embedding layer of LSTM.
                n_layers: Number of layers of LSTM.
                device: Pytorch device where the model should exist.
        """
        super().__init__()
        defaultkwargs = {"hidden_dim": 48, "n_layers": 3, "device": torch.device("cpu")}
        kwargs = {**defaultkwargs, **kwargs}
        self.n_dim_out = num_nodes * (num_nodes - 1)
        self.hidden_dim = kwargs["hidden_dim"]
        self.n_layers = kwargs["n_layers"]
        self.device = kwargs["device"]
        self.rnn = torch.nn.LSTM(self.hidden_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True).to(
            self.device
        )
        self.proj = torch.nn.Linear(self.hidden_dim, 1).to(self.device)
        self.embed = torch.nn.Linear(1, self.hidden_dim).to(self.device)

        # Create variables h0 and c0 which are the learnable hidden states of the LSTM

        self.h0 = torch.nn.Parameter(1e-3 * torch.randn(self.n_layers, 1, self.hidden_dim)).to(self.device)
        self.c0 = torch.nn.Parameter(1e-3 * torch.randn(self.n_layers, 1, self.hidden_dim)).to(self.device)
        # create variable for the initial input of the LSTM
        self._init_input_param = torch.nn.Parameter(torch.zeros(1, 1, 1)).to(self.device)

    def forward(self, token: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        """
        Forward function for accepting the inputs and state and then taking a forward pass through one rnn timestep.
        Args:
                inputs: Inputs of shape (batch_size, 1, 1) to lstm.
                state: The state vectors of lstm (c, h)
        Returns the logits (batch_size, 1, 1) of the Beroulli of the matrix and the updated states.
        """
        token = self.embed.forward(token)
        out, state = self.rnn.forward(token, state)
        logit = self.proj.forward(out)
        return logit, state

    def get_state(self, batch_size: int = 1):
        """
        Helper function for tiling the state vectors of lstm.
        """
        return (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1))

    def init_input(self, batch_size: int = 1):
        """
        Helper function for tiling the input to lstm.
        """
        return self._init_input_param.expand(batch_size, 1, 1)

    def vec_to_adj_mat(self, A_flat: torch.Tensor) -> torch.Tensor:
        """
        Convert a flat vector of non-diagonal adjacency elements to a full matrix with diagonals of 0 added.
        Args:
                A_flat: Flattened adjacency matrix without diagonal of shape (..., d(d-1))
                num_nodes: Number of nodes d.
        Returns the adjacency matrix of each flat vector with diagonal of 0's.
        """
        if len(A_flat.shape) == 1:
            A_flat = A_flat.unsqueeze(0)
        upper_triangular = fill_triangular(A_flat[..., : int(self.n_dim_out / 2)], upper=True)
        lower_triangular = fill_triangular(A_flat[..., : int(self.n_dim_out / 2)], upper=False)
        return upper_triangular + lower_triangular

    def adj_mat_to_vec(self, A: torch.Tensor) -> torch.Tensor:
        """
        Return a flattened vector corresponding to the adjacency matrix A with diagonals removed.
        """
        vec_upper = unfill_triangular(A, upper=True)
        vec_lower = unfill_triangular(A, upper=False)

        return torch.cat([vec_upper, vec_lower], dim=-1)


class AutoregressiveDistribution(AdjacencyDistribution):
    """
    A class that parameterises the distribution over the Bernoulli adjacency matrix as an autoregressive distribution
    over the entries.
    """

    def __init__(
        self, num_nodes: int, transition_function: Optional[Type[LSTM]] = None, num_mc_samples: int = 1000, **kwargs
    ):
        if transition_function is None:
            transition_function = LSTM
        self.transition_function = transition_function(num_nodes=num_nodes, **kwargs)
        self.num_mc_samples = num_mc_samples
        super().__init__(num_nodes, validate_args=False)

    def _sample(
        self,
        sample_shape: torch.Size,
        reparametrized: bool = False,
        hard: bool = False,
        temperature: float = 0.0,
    ):
        """
        Sample a binary adjacency matrix, with support for both discrete sampling as well as sampling from the Gumbel Simplex.
        Args:
                sample_shape: the shape of the samples to return
                reparameterized (bool): whether to perform reparameterized autoregressive sampling.
                hard (bool): Whether to threshold reparameterized samples using the straight-through trick.
                temperature (float): The temperature for the gumbel softmax distribution.
        Returns:
                A tensor of shape sample_shape + (num_nodes, num_nodes)
        """
        flattened_shape = int(np.prod(sample_shape))
        state = self.transition_function.get_state(flattened_shape)  # hidden / cell state at t=0
        token = self.transition_function.init_input(flattened_shape)  # input at t=0
        sampled_tokens = []  # Collect the logits of the adjacency matrix which is autoregressively sampled
        for _ in range(self.transition_function.n_dim_out):
            logits, state = self.transition_function.forward(token, state)
            if reparametrized:
                _sample = td.RelaxedBernoulli(temperature=temperature, logits=logits).rsample()
            else:
                _sample = td.Bernoulli(logits=logits).sample()
            token = _sample
            sampled_tokens.append(_sample)
        samples = torch.cat(sampled_tokens, dim=1)
        if reparametrized and hard:
            samples_hard = torch.round(samples)
            samples = (samples_hard - samples).detach() + samples
        return self.transition_function.vec_to_adj_mat(A_flat=samples.squeeze())

    def relaxed_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0) -> torch.Tensor:
        """
        Sample a binary adjacency matrix from the relaxed distribution.
        For this class this is the same as the sample method.
        Args:
                sample_shape: the shape of the samples to return
        Returns:
                A tensor of shape sample_shape + (num_nodes, num_nodes)
        """
        return self._sample(
            sample_shape=sample_shape,
            reparametrized=True,
            hard=True,
            temperature=temperature,
        )

    def relaxed_soft_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0) -> torch.Tensor:
        """
        Sample a binary adjacency matrix from the relaxed distribution without thresholding.
        For this class this is the same as the sample method.
        Args:
                sample_shape: the shape of the samples to return
        Returns:
                A tensor of shape sample_shape + (num_nodes, num_nodes)
        """
        return self._sample(
            sample_shape=sample_shape,
            reparametrized=True,
            hard=False,
            temperature=temperature,
        )

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample a binary adjacency matrix from the underlying distribution.
        Args:
                sample_shape: the shape of the samples to return
        Returns:
                A tensor of shape sample_shape + (num_nodes, num_nodes)
        """
        return self._sample(sample_shape=sample_shape, reparametrized=False, hard=True)

    def entropy(self) -> torch.Tensor:
        """
        Return the entropy of the underlying distribution.
        Returns:
                A tensor of shape (1), with the entropy of the distribution
        """
        return -torch.mean(self.log_prob(self.sample(torch.Size((self.num_mc_samples,)))))

    @property
    def mean(self) -> torch.Tensor:
        """
        Return the mean of the underlying distribution.
        This will be the adjacency matrix.
        Returns:
                A tensor of shape (num_nodes, num_nodes)
        """
        return torch.mean(self.sample(torch.Size((self.num_mc_samples,))), dim=0)

    @property
    def mode(self) -> torch.Tensor:
        """
        Return the mode of the underlying distribution.
        This will be the adjacency matrix.
        Returns:
                A tensor of shape (num_nodes, num_nodes)
        """
        samples = self.sample(torch.Size((self.num_mc_samples,)))
        best_sample_idx = torch.argmax(self.log_prob(samples))
        return samples[best_sample_idx]

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of each tensor from the sample space.
        Args:
                value: a binary matrix of shape (..., n, n)
        Returns:
                A tensor of shape (...), with the log probabilities of each tensor in the batch.
        """
        value_vec = self.transition_function.adj_mat_to_vec(value)
        sample_shape = value_vec.shape[:-1]
        value_ = value_vec.view(-1, self.num_nodes * (self.num_nodes - 1), 1).to(self.transition_function.device)
        state = self.transition_function.get_state(value_.shape[0])  # hidden / cell state at t=0
        init_value = self.transition_function.init_input(value_.shape[0])  # input at t=0
        value_input = torch.cat([init_value, value_], dim=-2)
        logits, _ = self.transition_function.forward(value_input, state)
        logits = logits[
            :, :-1, :
        ]  # Exclude the logits of the t+1 prediction (as it is not a logit corresponding to adj. matrix)
        value_vec = value_input[:, 1:, :]  # Remove the 0th token as it is just an init token
        log_probs = td.Bernoulli(logits=logits).log_prob(value_vec).sum((-2, -1))
        return log_probs.view(*sample_shape)
