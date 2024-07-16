from typing import Optional

import torch
import torch.distributions as td


class ExpertGraphContainer(torch.nn.Module):
    def __init__(self, dag: torch.Tensor, mask: torch.Tensor, confidence: float, scale: float) -> None:
        """Container holding an "experts" prior belief about the underlying causal DAG.

        Arguments:
            dag: The binary adjacency matrix representing domain knowledge in the form of a DAG. Corresponds to `mask`.
            mask: A binary mask indicating whether or not the corresponding edge of the `dag` has information.
            confidence: A value in the interval (0, 1] indicating the confidence of the existence of the edge
            scale: Scaling factor for expert graph loss
        """
        super().__init__()

        self.dag = torch.nn.Parameter(dag, requires_grad=False)
        self.mask = torch.nn.Parameter(mask, requires_grad=False)
        self.confidence: torch.Tensor
        self.scale: torch.Tensor
        self.register_buffer("confidence", torch.tensor(confidence, dtype=torch.float))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float))


class GibbsDAGPrior(td.Distribution):
    """
    Represents a prior distribution over adjacency matrices.

    The prior distribution consists of two terms:
        A sparsity term
        A Expert Graph term that represents some known prior belief about the graph.

    Each term has an associated parameter (lambda)

    The event shape of the prior is either (num_nodes, num_nodes) or (context_length, num_nodes, num_nodes) for temporal
    models and is set at initialization. The shape is checked for given matrices.
    """

    arg_constraints: dict = {}

    def __init__(
        self,
        num_nodes: int,
        sparsity_lambda: float,
        expert_graph_container: Optional[ExpertGraphContainer] = None,
        context_length: int | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            num_nodes: Number of nodes in the graph
            sparsity_lambda: Coefficient of sparsity term
            dagness_alpha: Coefficient of dagness term
            dagness_rho: Coefficient of squared dagness term
            expert_graph_container: Dataclass containing prior belief about the real graph
            context_length: Optional, specifying the context_length in temporal models
        """

        event_shape = (num_nodes, num_nodes) if context_length is None else (context_length, num_nodes, num_nodes)
        super().__init__(torch.Size(), event_shape=torch.Size(event_shape), **kwargs)

        if expert_graph_container is not None:
            if expert_graph_container.dag.shape != event_shape:
                raise ValueError(
                    f"Expert graph shape {expert_graph_container.dag.shape} does not match event shape {event_shape}"
                )
            if expert_graph_container.mask.shape != event_shape:
                raise ValueError(
                    f"Expert graph mask shape {expert_graph_container.mask.shape} does not match event shape {event_shape}"
                )

        self._num_nodes = num_nodes
        self._expert_graph_container = expert_graph_container
        self._sparsity_lambda = sparsity_lambda
        self._context_length = context_length

    def get_sparsity_term(self, A: torch.Tensor) -> torch.Tensor:
        """
        A term that encourages sparsity (see https://arxiv.org/pdf/2106.07635.pdf).
        The term is small when A is sparse.

        Args:
            A (torch.Tensor): Adjacency matrix of shape event_shape.

        Returns:
            Sparsity term.
        """
        return A.abs().sum()

    def get_expert_graph_term(self, A: torch.Tensor) -> torch.Tensor:
        """
        A term that encourages A to be close to given expert graph.

        Args:
            A (torch.Tensor): Adjacency matrix of shape event_shape.

        Returns:
            (torch.Tensor): Expert graph term.

        """
        assert isinstance(self._expert_graph_container, ExpertGraphContainer)
        assert A.shape[-len(self.event_shape) :] == self.event_shape
        return (
            (
                self._expert_graph_container.mask
                * (A - self._expert_graph_container.confidence * self._expert_graph_container.dag)
            )
            .abs()
            .sum()
        )

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Calculates the (un-normalized) log probability of adjacency matrix `value`
        under the distribution given by this instance.

        Args:
            value (torch.Tensor): Adjacency matrix of shape event_shape.

        Returns:
            (torch.Tensor): The un-normalized log probability of `value`.

        """
        assert value.shape[-len(self.event_shape) :] == self.event_shape

        log_prob = -self._sparsity_lambda * self.get_sparsity_term(value)

        if self._expert_graph_container is not None:
            log_prob -= self._expert_graph_container.scale * self.get_expert_graph_term(value)

        return log_prob
