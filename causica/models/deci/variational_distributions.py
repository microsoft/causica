from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import nn

from ...utils.causality_utils import admg2dag, dag2admg


class AdjMatrix(ABC):
    """
    Adjacency matrix interface for DECI
    """

    @abstractmethod
    def get_adj_matrix(self, do_round: bool = True) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        raise NotImplementedError()

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of distribution q. In this case 0.
        """
        raise NotImplementedError()

    @abstractmethod
    def sample_A(self) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        raise NotImplementedError()


class VarDistA(AdjMatrix, nn.Module):
    """
    Abstract class representing a variational distribution over binary adjacency matrix.
    """

    def __init__(
        self,
        device: torch.device,
        input_dim: int,
        tau_gumbel: float = 1.0,
    ):
        """
        Args:
            device: Device used.
            input_dim: dimension.
            tau_gumbel: temperature used for gumbel softmax sampling.
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.tau_gumbel = tau_gumbel

    @abstractmethod
    def _get_logits_softmax(self) -> torch.Tensor:
        """
        Returns the (softmax) logits.
        """
        raise NotImplementedError()

    def _build_bernoulli(self) -> td.Distribution:
        """
        Builds and returns the bernoulli distributions obtained using the (softmax) logits.
        """
        logits = self._get_logits_softmax()  # (2, n, n)
        logits_bernoulli_1 = logits[1, :, :] - logits[0, :, :]  # (n, n)
        # Diagonal elements are set to 0
        logits_bernoulli_1 -= 1e10 * torch.eye(self.input_dim, device=self.device)
        dist = td.Independent(td.Bernoulli(logits=logits_bernoulli_1), 2)
        return dist

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of distribution q.
        """
        return self._build_bernoulli().entropy()

    def sample_A(self) -> torch.Tensor:
        """
        Sample an adjacency matrix from the variational distribution. It uses the gumbel_softmax trick,
        and returns hard samples (straight through gradient estimator). Adjacency returned always has
        zeros in its diagonal (no self loops).

        V1: Returns one sample to be used for the whole batch.
        """
        logits = self._get_logits_softmax()
        sample = F.gumbel_softmax(logits, tau=self.tau_gumbel, hard=True, dim=0)  # (2, n, n) binary
        sample = sample[1, :, :]  # (n, n)
        sample = sample * (1 - torch.eye(self.input_dim, device=self.device))  # Force zero diagonals
        return sample

    def log_prob_A(self, A: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the variational distribution q(A) at a sampled adjacency A.

        Args:
            A: A binary adjacency matrix, size (input_dim, input_dim).

        Returns:
            The log probability of the sample A. A number if A has size (input_dim, input_dim).
        """
        return self._build_bernoulli().log_prob(A)

    @abstractmethod
    def get_adj_matrix(self, do_round: bool = True) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        raise NotImplementedError()


class DeterministicAdjacency(AdjMatrix):
    """
    Deterministic adjacency matrix used for DECI + true Graph
    """

    def __init__(
        self,
        device: torch.device,
    ):
        """
        Args:
            device: Device used.
        """
        self.adj_matrix: Optional[torch.Tensor] = None
        self.device = device

    def set_adj_matrix(self, adj_matrix: np.ndarray) -> None:
        """
        Set fixed adjacency matrix
        """
        self.adj_matrix = nn.Parameter(torch.from_numpy(adj_matrix).to(self.device), requires_grad=False)

    def get_adj_matrix(self, do_round: bool = True) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        assert self.adj_matrix is not None
        return self.adj_matrix

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of distribution q. In this case 0.
        """
        return torch.zeros(1, device=self.device)

    def sample_A(self) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        assert self.adj_matrix is not None
        return self.adj_matrix


class VarDistA_Simple(VarDistA):
    """
    Variational distribution for the binary adjacency matrix. Parameterizes the probability of each edge
    (including orientation).
    """

    def __init__(
        self,
        device: torch.device,
        input_dim: int,
        tau_gumbel: float = 1.0,
    ):
        """
        Args:
            device: Device used.
            input_dim: dimension.
            tau_gumbel: temperature used for gumbel softmax sampling.
        """
        super().__init__(device, input_dim, tau_gumbel)
        self.logits = self._initialize_params()

    def _initialize_params(self) -> torch.Tensor:
        """
        Returns the initial logits to sample A, a tensor of shape (2, input_dim, input_dim).
        Right now initialize all to zero. Could change. Could also change parameterization
        to be similar to the paper Cheng sent (https://arxiv.org/pdf/2107.10483.pdf).
        """
        logits = torch.zeros(2, self.input_dim, self.input_dim, device=self.device)  # Shape (2, input_dim, input_dim)
        return nn.Parameter(logits, requires_grad=True)

    def _get_logits_softmax(self) -> torch.Tensor:
        """
        Returns the (softmax) logits.
        """
        return self.logits

    def get_adj_matrix(self, do_round: bool = True) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        probs_1 = F.softmax(self.logits, dim=0)[1, :, :]  # Shape (input_dim, input_dim)
        probs_1 *= 1 - torch.eye(self.input_dim, device=self.device)
        if do_round:
            return probs_1.round()
        return probs_1

    def get_print_params(self):
        """
        Will go away, returs parameters to print.
        """
        return self.logits


class VarDistA_ENCO(VarDistA):
    """
    Variational distribution for the binary adjacency matrix, following the parameterization from
    the ENCO paper (https://arxiv.org/pdf/2107.10483.pdf). For each edge, parameterizes the existence
    and orientation separately. Main benefit is that it avoids length 2 cycles automatically.
    Orientation is somewhat over-parameterized.
    """

    def __init__(self, device: torch.device, input_dim: int, tau_gumbel: float = 1.0, dense_init: bool = False):
        """
        Args:
            device: Device used.
            input_dim: dimension.
            tau_gumbel: temperature used for gumbel softmax sampling.
        """
        super().__init__(device, input_dim, tau_gumbel)
        self.dense_init = dense_init
        self.logits_edges = self._initialize_edge_logits()
        self.params_orient = self._initialize_orient_params()

    def _initialize_edge_logits(self) -> torch.Tensor:
        """
        Returns the initial logits that characterize the presence of an edge (gamma in the ENCO paper),
        a tensor of shape (2, n, n).
        """
        logits = torch.zeros(2, self.input_dim, self.input_dim, device=self.device)  # Shape (2, input_dim, input_dim)
        if self.dense_init:
            logits[1, :, :] += 3
        else:
            logits[1, :, :] -= 1
        return nn.Parameter(logits, requires_grad=True)

    def _initialize_orient_params(self) -> torch.Tensor:
        """
        Returns the initial logits that characterize the orientation (theta in the ENCO paper),
        a tensor of shape (n, n). Right now initialize all to zero. Could change.
        This will be processed so as to keep only strictly upper triangular, so some of
        these parameters are not trained.
        """
        if self.dense_init:
            params = torch.ones(self.input_dim, self.input_dim, device=self.device)  # (n, n)
        else:
            params = torch.zeros(self.input_dim, self.input_dim, device=self.device)  # (n, n)
        return nn.Parameter(params, requires_grad=True)

    def _build_logits_orient(self) -> torch.Tensor:
        """
        Auxiliary function that computes the (softmax) logits to sample orientation for the edges given the parameters.
        """
        logits_0 = torch.zeros(self.input_dim, self.input_dim, device=self.device)  # Shape (input_dim, input_dim)
        # Get logits_1 strictly upper triangular
        logits_1 = torch.triu(self.params_orient)
        logits_1 = logits_1 * (1.0 - torch.eye(self.input_dim, self.input_dim, device=self.device))
        logits_1 = logits_1 - torch.transpose(logits_1, 0, 1)  # Make logit_ij = -logit_ji
        return torch.stack([logits_0, logits_1])

    def _get_logits_softmax(self) -> torch.Tensor:
        """
        Auxiliary function to compute the (softmax) logits from both edge logits and orientation logits. Notice
        the logits for the softmax are computed differently than those for Bernoulli (latter uses sigmoid, equivalent
        if the logits for zero filled with zeros).

        Simply put, to sample an edge i->j you need to both sample the precense of that edge, and sample its orientation.
        """
        logits_edges = self.logits_edges  # Shape (2, input_dim, input_dim)
        logits_orient = self._build_logits_orient()  # Shape (2, input_dim, input_dim)
        logits_1 = logits_edges[1, :, :] + logits_orient[1, :, :]  # Shape (input_dim, input_dim)
        aux = torch.stack(
            [
                logits_edges[1, :, :] + logits_orient[0, :, :],
                logits_edges[0, :, :] + logits_orient[1, :, :],
                logits_edges[0, :, :] + logits_orient[0, :, :],
            ]
        )  # Shape (3, input_dim, input_dim)
        logits_0 = torch.logsumexp(aux, dim=0)  # Shape (input_dim, input_dim)
        logits = torch.stack([logits_0, logits_1])  # Shape (2, input_dim, input_dim)
        return logits

    def get_adj_matrix(self, do_round: bool = True) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        probs_edges = F.softmax(self.logits_edges, dim=0)[1, :, :]  # Shape (input_dim, input_dim)
        logits_orient = self._build_logits_orient()
        probs_orient = F.softmax(logits_orient, dim=0)[1, :, :]  # Shape (input_dim, input_dim)
        probs_1 = probs_edges * probs_orient
        probs_1 = probs_1 * (1.0 - torch.eye(self.input_dim, device=self.device))
        if do_round:
            return probs_1.round()
        return probs_1

    def get_print_params(self):
        """
        Will go away, returs parameters to print.
        """
        return self.logits_edges, self.params_orient


class ThreeWayGraphDist(AdjMatrix, nn.Module):
    """
    An alternative variational distribution for graph edges. For each pair of nodes x_i and x_j
    where i < j, we sample a three way categorical C_ij. If C_ij = 0, we sample the edge
    x_i -> x_j, if C_ij = 1, we sample the edge x_j -> x_i, and if C_ij = 2, there is no
    edge between these nodes. This variational distribution is faster to use than ENCO
    because it avoids any calls to `torch.stack`.

    Sampling is performed with `torch.gumbel_softmax(..., hard=True)` to give
    binary samples and a straight-through gradient estimator.
    """

    def __init__(
        self,
        device: torch.device,
        input_dim: int,
        tau_gumbel: float = 1.0,
    ):
        """
        Args:
            device: Device used.
            input_dim: dimension.
            tau_gumbel: temperature used for gumbel softmax sampling.
        """
        super().__init__()
        # We only use n(n-1)/2 random samples
        # For each edge, sample either A->B, B->A or no edge
        # We convert this to a proper adjacency matrix using torch.tril_indices
        self.logits = nn.Parameter(
            torch.zeros(3, (input_dim * (input_dim - 1)) // 2, device=device), requires_grad=True
        )
        self.tau_gumbel = tau_gumbel
        self.input_dim = input_dim
        self.device = device
        self.lower_idxs = torch.unbind(
            torch.tril_indices(self.input_dim, self.input_dim, offset=-1, device=self.device), 0
        )

    def _triangular_vec_to_matrix(self, vec):
        """
        Given an array of shape (k, n(n-1)/2) where k in {2, 3}, creates a matrix of shape
        (n, n) where the lower triangular is filled from vec[0, :] and the upper
        triangular is filled from vec[1, :].
        """
        output = torch.zeros((self.input_dim, self.input_dim), device=self.device)
        output[self.lower_idxs[0], self.lower_idxs[1]] = vec[0, ...]
        output[self.lower_idxs[1], self.lower_idxs[0]] = vec[1, ...]
        return output

    def get_adj_matrix(self, do_round: bool = False) -> torch.Tensor:
        """
        Returns the adjacency matrix of edge probabilities.
        """
        probs = F.softmax(self.logits, dim=0)  # (3, n(n-1)/2) probabilities
        out_probs = self._triangular_vec_to_matrix(probs)
        if do_round:
            return out_probs.round()
        else:
            return out_probs

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of distribution q, which is a collection of n(n-1) categoricals on 3 values.
        """
        dist = td.Categorical(logits=self.logits.transpose(0, -1))
        entropies = dist.entropy()
        return entropies.sum()

    def sample_A(self) -> torch.Tensor:
        """
        Sample an adjacency matrix from the variational distribution. It uses the gumbel_softmax trick,
        and returns hard samples (straight through gradient estimator). Adjacency returned always has
        zeros in its diagonal (no self loops).

        V1: Returns one sample to be used for the whole batch.
        """
        sample = F.gumbel_softmax(self.logits, tau=self.tau_gumbel, hard=True, dim=0)  # (3, n(n-1)/2) binary
        return self._triangular_vec_to_matrix(sample)


class TemporalThreeWayGrahpDist(ThreeWayGraphDist):
    """
    This class adapts the ThreeWayGraphDist s.t. it supports the variational distributions for temporal adjacency matrix.

    The principle is to follow the logic as ThreeWayGraphDist. The implementation has two separate part:
    (1) categorical distribution for instantaneous adj (see ThreeWayGraphDist); (2) Bernoulli distribution for lagged
    adj. Note that for lagged adj, we do not need to follow the logic from ThreeWayGraphDist, since lagged adj allows diagonal elements
    and does not have to be a DAG. Therefore, it is simpler to directly model it with Bernoulli distribution.
    """

    def __init__(
        self,
        device: torch.device,
        input_dim: int,
        lag: int,
        tau_gumbel: float = 1.0,
        init_logits: Optional[List[float]] = None,
    ):
        """
        This creates an instance of variational distribution for temporal adjacency matrix.
        Args:
            device: Device used.
            input_dim: The number of nodes for adjacency matrix.
            lag: The lag for the temporal adj matrix. The adj matrix has the shape (lag+1, num_nodes, num_nodes).
            tau_gumbel: The temperature for the gumbel softmax sampling.
            init_logits: The initialized logits value. If None, then use the default initlized logits (value 0). Otherwise,
            init_logits[0] indicates the non-existence edge logit for instantaneous effect, and init_logits[1] indicates the
            non-existence edge logit for lagged effect. E.g. if we want a dense initialization, one choice is (-7, -0.5)
        """
        # Call parent init method, this will init a self.logits parameters for instantaneous effect.
        super().__init__(device=device, input_dim=input_dim, tau_gumbel=tau_gumbel)
        # Create a separate logit for lagged adj
        # The logits_lag are initialized to zero with shape (2, lag, input_dim, input_dim).
        # logits_lag[0,...] indicates the logit prob for no edges, and logits_lag[1,...] indicates the logit for edge existence.
        self.lag = lag
        # Assertion lag > 0
        assert lag > 0
        self.logits_lag = nn.Parameter(torch.zeros((2, lag, input_dim, input_dim), device=device), requires_grad=True)
        self.init_logits = init_logits
        # Set the init_logits if not None
        if self.init_logits is not None:
            self.logits.data[2, ...] = self.init_logits[0]
            self.logits_lag.data[0, ...] = self.init_logits[1]

    def get_adj_matrix(self, do_round: bool = False) -> torch.Tensor:
        """
        This returns the temporal adjacency matrix of edge probability.
        Args:
            do_round: Whether to round the edge probabilities.

        Returns:
            The adjacency matrix with shape [lag+1, num_nodes, num_nodes].
        """

        # Create the temporal adj matrix
        probs = torch.zeros(self.lag + 1, self.input_dim, self.input_dim, device=self.device)
        # Generate simultaneous adj matrix
        probs[0, ...] = super().get_adj_matrix(do_round=do_round)  # shape (input_dim, input_dim)
        # Generate lagged adj matrix
        probs[1:, ...] = F.softmax(self.logits_lag, dim=0)[1, ...]  # shape (lag, input_dim, input_dim)
        if do_round:
            return probs.round()
        else:
            return probs

    def entropy(self) -> torch.Tensor:
        """
        This computes the entropy of the variational distribution. This can be done by (1) compute the entropy of instantaneous adj matrix(categorical, same as ThreeWayGraphDist),
        (2) compute the entropy of lagged adj matrix (Bernoulli dist), and (3) add them together.
        """
        # Entropy for instantaneous dist, call super().entropy
        entropies_inst = super().entropy()

        # Entropy for lagged dist
        # batch_shape [lag], event_shape [num_nodes, num_nodes]
        dist_lag = td.Independent(td.Bernoulli(logits=self.logits_lag[1, ...] - self.logits_lag[0, ...]), 2)
        entropies_lag = dist_lag.entropy().sum()
        return entropies_lag + entropies_inst

    def sample_A(self) -> torch.Tensor:
        """
        This samples the adjacency matrix from the variational distribution. This uses the gumbel softmax trick and returns
        hard samples. This can be done by (1) sample instantaneous adj matrix using self.logits, (2) sample lagged adj matrix using self.logits_lag.
        """

        # Create adj matrix to avoid concatenation
        adj_sample = torch.zeros(
            self.lag + 1, self.input_dim, self.input_dim, device=self.device
        )  # shape (lag+1, input_dim, input_dim)

        # Sample instantaneous adj matrix
        adj_sample[0, ...] = self._triangular_vec_to_matrix(
            F.gumbel_softmax(self.logits, tau=self.tau_gumbel, hard=True, dim=0)
        )  # shape (input_dim, input_dim)
        # Sample lagged adj matrix
        adj_sample[1:, ...] = F.gumbel_softmax(self.logits_lag, tau=self.tau_gumbel, hard=True, dim=0)[
            1, ...
        ]  # shape (lag, input_dim, input_dim)
        return adj_sample


class VarDistA_ENCO_ADMG(VarDistA_ENCO):
    """Variational distribution for an acyclic directed mixed graph (ADMG).

    A variational distribution over two adjacency matrices, the first describes directed edges and the latter bidirected
    edges between observed variables.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params_bidirected = self._initialize_bidirected_params()

    def _initialize_bidirected_params(self) -> torch.Tensor:
        """Initialises logits that characterise bidirectional edges between observed variables."""
        if self.dense_init:
            params = torch.ones(self.input_dim, self.input_dim, device=self.device)
        else:
            params = torch.zeros(self.input_dim, self.input_dim, device=self.device)

        return nn.Parameter(params, requires_grad=True)

    def _build_logits_bidirected(self) -> torch.Tensor:
        """Auxiliary function to build the logits to sample the bidirected edges."""
        logits_0 = torch.zeros(self.input_dim, self.input_dim, device=self.device)
        # logits_1 is stricly upper triangular.
        logits_1 = torch.triu(self.params_bidirected)
        logits_1 = logits_1 * (1.0 - torch.eye(self.input_dim, self.input_dim, device=self.device))
        logits_1 = logits_1 + torch.transpose(logits_1, 0, 1)  # Make logit_ij = logit_ji
        return torch.stack([logits_0, logits_1])

    def get_bidirected_adj_matrix(self, do_round: bool = False) -> torch.Tensor:
        """Returns the bidirected adjacency matrix."""
        probs = F.softmax(self._build_logits_bidirected(), dim=0)[1, :, :]
        probs = probs * (1.0 - torch.eye(self.input_dim, device=self.device))
        if do_round:
            return probs.round()
        return probs

    def get_directed_adj_matrix(self, do_round: bool = False) -> torch.Tensor:
        """Returns the directed adjacency matrix."""
        return super().get_adj_matrix(do_round)

    def get_adj_matrix(self, do_round: bool = False) -> torch.Tensor:
        """Returns the adjacency matrix over both observed and latent variables."""
        directed_adj = self.get_directed_adj_matrix(do_round)
        bidirected_adj = self.get_bidirected_adj_matrix(do_round)
        return self.magnify_adj_matrices(directed_adj, bidirected_adj)

    def _build_bidirected_bernoulli(self) -> td.Distribution:
        """Builds the Bernoulli distribution obtained using the logits."""
        logits = self._build_logits_bidirected()
        logits_bernoulli_1 = logits[1, :, :] - logits[0, :, :]
        # Diagonal elements are set to 0
        logits_bernoulli_1 -= 1e10 * torch.eye(self.input_dim, device=self.device)
        dist = td.Independent(td.Bernoulli(logits=logits_bernoulli_1), 2)
        return dist

    def sample_A(self) -> torch.Tensor:
        """Samples the directed and bidirected matrix from the variational distribution and returns the corresponding
        adjacency matrix over both the observed and latent varaibles."""
        directed_adj_sample = self.sample_directed_adj()
        bidirected_adj_sample = self.sample_bidirected_adj()

        return self.magnify_adj_matrices(directed_adj_sample, bidirected_adj_sample)

    def entropy(self) -> torch.Tensor:
        """Computes the entropy of the variational distribution."""
        # Distribution is only over half the bidirected adjacency matrix.
        return self._build_bernoulli().entropy() + 0.5 * self._build_bidirected_bernoulli().entropy()

    def sample_directed_adj(self) -> torch.Tensor:
        """Samples a directed adjacency matrix from the variational distribution.

        Samples a directed adjacency matrix from the variational distribution using the gumbel softmax trick and
        returns hard samples (straight through the gradient estimator). Adjacency returned always has zeros in the
        diagonal.
        """
        return super().sample_A()

    def sample_bidirected_adj(self) -> torch.Tensor:
        """Samples a bidirected adjacency matrix from the variational distribution.

        Samples a bidirected adjacency matrix from the variational distribution using the gumbel softmax trick and
        returns hard samples (straight through the gradient estimator). Adjacency returned is symmetric and always has
        zeros in the diagonal.
        """
        logits = self._build_logits_bidirected()
        sample = F.gumbel_softmax(logits, tau=self.tau_gumbel, hard=True, dim=0)  # (2, n, n) binary
        sample = torch.triu(sample[1, :, :])  # (n, n)
        sample = sample * (1 - torch.eye(self.input_dim, device=self.device))  # Force zero diagonals
        sample = sample + torch.transpose(sample, 0, 1)  # Force symmetry
        return sample

    def log_prob_bidirected(self, bidirected_adj: torch.Tensor) -> torch.Tensor:
        """Evaluates the variational distribution the sampled bidirectional adjacency matrix.

        Args:
            bidirected_adj: Bidirectional adjacency matrix.

        Returns:
            The log probability of the sample.
        """
        return self._build_bidirected_bernoulli().log_prob(bidirected_adj)

    def log_prob_directed(self, directed_adj: torch.Tensor) -> torch.Tensor:
        """Evaluates the variational distribution at the sampled directional adjacency matrix.

        Args:
            directed_adj: Directional adjacency matrix.

        Returns:
            The log probability of the sample.
        """
        return self._build_bernoulli().log_prob(directed_adj)

    def log_prob_A(self, A: torch.Tensor) -> torch.Tensor:
        """Evaluates the variational distribution at a samples adjacency A.

        Args:
            A: A binary adjacency matrix, size (input_dim + latent_dim, input_dim + latent_dim).

        Returns:
            The log probability of the sample A. A number if A has size (input_dim + latent_dim, input_dim + latent_dim).
        """
        directed_adj, bidirected_adj = self.demagnify_adj_matrix(A)
        return self.log_prob_directed(directed_adj) + self.log_prob_bidirected(bidirected_adj)

    def magnify_adj_matrices(
        self,
        directed_adj: torch.Tensor,
        bidirected_adj: torch.Tensor,
    ) -> torch.Tensor:
        """Magnifies the two adjacency matrices to create a larger adjacency matrix over both oberved and latent
        variables.

        Args:
            directed_adj: Directed adjacency matrix over the observed variables.
            bidirected_adj: Bidirected adjacency matrix over the observed variables.

        Returns:
            Magnified adjacency matrix.
        """
        return admg2dag(directed_adj, bidirected_adj)

    def demagnify_adj_matrix(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Demagnifies the adjacency matrix over both observed and latent variables to create a directed and
        bidirected adjacency matrix.

        Args:
            adj: The adjacency matrix over both oberserved and latent variables.

        Returns:
            Tuple[torch.Tensor, torch.Tensor] containing the directed and bidirected adjacency matrix.
        """
        return dag2admg(adj)


class CategoricalAdjacency(AdjMatrix, nn.Module):
    """Class representing a uniform categorical distribution over multiple adjacency matrices."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.adj_matrices: Optional[torch.Tensor] = None
        self.device = device

    def set_adj_matrices(self, adj_matrices: np.ndarray) -> None:
        self.adj_matrices = nn.Parameter(
            torch.from_numpy(adj_matrices.astype(np.float32)).to(self.device), requires_grad=False
        )

    def _build_categorical(self) -> td.Distribution:
        assert self.adj_matrices is not None

        dist = td.Categorical(logits=torch.ones(self.adj_matrices.shape[0], device=self.device))
        return dist

    def entropy(self) -> torch.Tensor:
        return self._build_categorical().entropy()

    def sample_A(self) -> torch.Tensor:
        assert self.adj_matrices is not None

        return self.adj_matrices[self._build_categorical().sample()]

    def log_prob_A(self, A: torch.Tensor) -> torch.Tensor:
        assert self.adj_matrices is not None
        assert any((torch.isclose(A, adj).all() for adj in self.adj_matrices)), "log probability of negative infinity"

        return torch.log(torch.as_tensor(1 / self.adj_matrices.shape[0]))

    def get_adj_matrix(self, do_round: bool = True) -> torch.Tensor:
        assert self.adj_matrices is not None
        return self.adj_matrices[0]
