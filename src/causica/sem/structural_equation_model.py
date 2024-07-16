import abc
from typing import Iterable, Optional, Sequence

import torch
import torch.distributions as dist
from tensordict import TensorDict


class SEM(dist.Distribution, abc.ABC):
    """A structural equation model (SEM).

    A SEM defines the causal relationships amongst a given set of nodes. This class provides methods to sample data
    from the observational and interventional distributions of the SEM.

    A SEMs have a batch shape: `batch_shape_f + batch_shape_g`

    In general, a SEM has a batch shape corresponding to `functions shape + graphs shape` specified as `batch_shape_f`
    and `batch_shape_g` respectively. This allows for batched processing of different functions (e.g. interventions on
    the same nodes) on different graphs. Samples and noise are broadcast to include this shape before calling the
    functional relationships.
    """

    arg_constraints: dict = {}

    def __init__(
        self,
        graph: torch.Tensor,
        node_names: Sequence[str],
        event_shape: torch.Size = torch.Size(),
        batch_shape: torch.Size | None = None,
    ) -> None:
        """
        Args:
            graph: Adjacency Matrix for the underlying SEM
            node_names: The names of each node in the graph
            event_shape: Shape of a sample from the distribution. This is currently not used because the output shape is
                determined by the dictionary holding the samples.
            batch_shape: Batch shape of the distribution. This decomposes into the batch shape of the functions and the
                batch shape of the graphs: batch_shape = batch_shape_f + batch_shape_g. If None, the batch shape of the
                graph is used.
        """
        self._graph = graph
        self.node_names = node_names
        assert node_names, "Can't have an empty SEM"
        assert graph.shape[-2:] == (
            len(node_names),
            len(node_names),
        ), "Graph adjacency matrix must have shape [num_nodes, num_nodes] (excluding batch dimensions)"
        if batch_shape is None:
            self.batch_shape_g = torch.Size(graph.shape[:1]) if graph.ndim > 2 else torch.Size([])
            batch_shape = self.batch_shape_g
            self.batch_shape_f = torch.Size([])
        super().__init__(event_shape=event_shape, batch_shape=batch_shape)

    @property
    def graph(self) -> torch.Tensor:
        """
        The adjacency matrix representing the graph.
        """
        return self._graph

    @abc.abstractmethod
    def log_prob(self, value: TensorDict) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def do(self, interventions: TensorDict) -> "SEM":
        """Return the SEM associated with the interventions.

        If the interventions have a batch shape, then the returned functions will have the same batch shape to batch
        the computation across interventions.
        """

    def condition(self, conditioning: TensorDict) -> "SEM":
        """Return a SEM with conditioned variables.

        This implementation of conditioning relies on the do operator and only supports conditioning on a conditioning
        set with no unconditioned parents. This is a limitation of the current implementation and will be lifted in the
        future. For example in a SEM with the following graph: A -> B -> C and A -> C, conditioning on B is not
        supported yet. However, conditioning on A or {A, B} is supported.

        Args:
            conditioning: A dictionary of conditioning values for the nodes in the SEM. There must not be any nodes
                outside of the conditioning set that are parents of the conditioning set.

        Return:
            The SEM conditioned on the conditioning set
        """
        conditioning_node_names = set(conditioning.keys())
        unconditioned_node_names = [
            name for name in self.node_names if name not in conditioning_node_names
        ]  # keep order

        unconditioned_idx = [self.node_names.index(name) for name in unconditioned_node_names]

        for i, name in enumerate(self.node_names):
            if name in conditioning_node_names and self.graph[unconditioned_idx, i].sum() > 0:
                parents = set(node for idx, node in enumerate(self.node_names) if self.graph[idx, i] > 0)
                unconditioned_parents = parents - conditioning_node_names
                raise NotImplementedError(
                    f"Conditioning on a node with unconditioned parents is not supported yet. Tried conditioning on {name} with parents {unconditioned_parents}"
                )

        return self.do(conditioning)

    @abc.abstractmethod
    def noise_to_sample(self, noise: TensorDict) -> TensorDict:
        """
        For a given noise vector, return the corresponding sample values.

        Args:
            noise: Dictionary of tensors representing the noise shape sample_shape + batch_shape + noise_shape
        Return:
            Dictionary of samples from the sem corresponding to the noise, shape sample_shape + batch_shape + event_shape
        """

    @abc.abstractmethod
    def sample_to_noise(self, sample: TensorDict) -> TensorDict:
        """
        For a given sample get the noise vector.

        Args:
            sample: samples from the sem each of shape sample_shape + batch_shape + event_shape
        Return:
            Dictionary of tensors representing the noise shape sample_shape + batch_shape + noise_shape
        """

    @torch.no_grad()
    def sample(self, sample_shape: torch.Size = torch.Size()) -> TensorDict:
        """
        Sample from the SEM

        Grads shall not pass through this method (see `Distribution.sample` vs `Distribution.rsample`).

        Args:
            sample_shape: shape of the returned samples
        Return:
            Dictionary of samples from the sem corresponding to the noise, shape sample_shape + batch_shape + event_shape
        """
        noise_dict = self.sample_noise(sample_shape=sample_shape)
        return self.noise_to_sample(noise=noise_dict)

    @abc.abstractmethod
    @torch.no_grad()
    def sample_noise(self, sample_shape: torch.Size = torch.Size()) -> TensorDict:
        """
        Sample the noise vector for the distribution.

        Grads shall not pass through implementations of this method.

        Args:
            sample_shape: shape of the returned noise samples
        Return:
            Dictionary of samples from the noise distribution for each node, shape sample_shape + batch_shape + event_shape
        """


def ite(
    sem: SEM,
    factual_data: TensorDict,
    intervention_a: TensorDict,
    intervention_b: TensorDict,
    effects: Optional[Iterable[str]] = None,
) -> TensorDict:
    """Calculate ITE of intervention A and B on some factual data for a list of effects.

    Args:
        factual_data: Factual data to abduct the noise from.
        intervention_a: Specification of intervention A.
        intervention_b: Specification of intervention B.
        effects: List of effect variables. Defaults to None.

    Returns:
       TensorDict: Dictionary holding the ITEs for the effect variables for all samples.
    """
    if effects is None:
        effects = list(set(sem.node_names) - set(intervention_a.keys()) - set(intervention_b.keys()))

    base_noise = sem.sample_to_noise(factual_data)
    do_a_cfs = sem.do(interventions=intervention_a).noise_to_sample(base_noise)
    do_b_cfs = sem.do(interventions=intervention_b).noise_to_sample(base_noise)

    return {effect: do_a_cfs[effect] - do_b_cfs[effect] for effect in effects}


def counterfactual(sem: SEM, factual_data: TensorDict, intervention: TensorDict) -> TensorDict:
    """Calculate Counterfactual of an intervention on some factual data.

    Args:
        factual_data: Factual data to abduct the noise from.
        intervention: Specification of intervention.

    Returns:
       TensorDict: Dictionary holding the counterfactual values for all samples.
    """
    return sem.do(interventions=intervention).noise_to_sample(sem.sample_to_noise(factual_data))


def ate(
    sem: SEM,
    intervention_a: TensorDict,
    intervention_b: TensorDict,
    effects: Optional[Iterable[str]] = None,
    num_samples: int = 1000,
) -> TensorDict:
    """Calculate the ATE of intervention A and B for a list of effects.

    Args:
        intervention_a: Specification of intervention A.
        intervention_b: Specification of intervention B.
        effects: List of effect variables. Defaults to None.
        num_samples: Number of Monte-Carlo samples to estimate the ATE. Defaults to 1000.

    Returns:
       TensorDict: Dictionary holding the ATEs for the effect variables.
    """
    sample_shape = torch.Size([num_samples])

    if effects is None:
        effects = list(set(sem.node_names) - set(intervention_a.keys()) - set(intervention_b.keys()))

    do_a_samples = sem.do(interventions=intervention_a).sample(sample_shape)
    do_b_samples = sem.do(interventions=intervention_b).sample(sample_shape)

    return TensorDict(
        {effect: do_a_samples[effect].mean(0) - do_b_samples[effect].mean(0) for effect in effects}, batch_size=tuple()
    )
