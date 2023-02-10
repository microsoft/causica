import abc
from typing import Optional

import torch
import torch.distributions as td


class AdjacencyDistribution(td.Distribution, abc.ABC):
    """
    Probability distributions over binary adjacency matrices for graphs.

    NOTE: Because we want to differentiate through samples (which are binary matrices),
    we add a `relaxed_sample` method. This method is still expected to return valid samples (binary matrices).

    Since the statistics of the relaxed distribution can be difficult to calculate,
    we choose to implement `relaxed_sample` as the relaxed distribution but report other statistics for
    the underlying distribution, as well as providing the usual `sample` method for it.

    Clearly this approximation could be invalid when the relaxed distribution is very different from
    the underlying one (at high temperatures).
    """

    support = td.constraints.independent(td.constraints.boolean, 1)

    def __init__(self, num_nodes: int, validate_args: Optional[bool] = None):
        assert num_nodes > 0, "Number of nodes in the graph must be greater than 0"
        self.num_nodes = num_nodes
        event_shape = torch.Size((num_nodes, num_nodes))

        super().__init__(event_shape=event_shape, validate_args=validate_args)

    @abc.abstractmethod
    def relaxed_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0) -> torch.Tensor:
        """
        Sample a binary adjacency matrix from the relaxed distribution (see NOTE in the class docstring).

        Args:
            sample_shape: the shape of the samples to return
            temperature: The temperature of the relaxed distribution
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Sample a binary adjacency matrix from the underlying distribution.

        Args:
            sample_shape: the shape of the samples to return
        Returns:
            A tensor of shape sample_shape + batch_shape + (num_nodes, num_nodes)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def entropy(self) -> torch.Tensor:
        """
        Return the entropy of the underlying distribution.

        Returns:
            A tensor of shape batch_shape, with the entropy of the distribution
        """

        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mean(self) -> torch.Tensor:
        """
        Return the mean of the underlying distribution.

        This will be a matrix with all entries in the interval [0, 1].

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def mode(self) -> torch.Tensor:
        """
        Return the mode of the underlying distribution.

        This will be an adjacency matrix.

        Returns:
            A tensor of shape batch_shape + (num_nodes, num_nodes)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Get the log probability of each tensor from the sample space

        Args:
            value: a binary matrix of shape value_shape + batch_shape + (n, n)
        Returns:
            A tensor of shape value_shape + batch_shape, with the log probabilities of each tensor in the batch.
        """
        raise NotImplementedError
