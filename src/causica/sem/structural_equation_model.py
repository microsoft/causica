import abc
from typing import Optional, Sequence

import torch
import torch.distributions as dist
from tensordict import TensorDict


class SEM(dist.Distribution, abc.ABC):
    arg_constraints: dict = {}

    def __init__(
        self,
        graph: torch.Tensor,
        node_names: Sequence[str],
        event_shape: torch.Size = torch.Size(),
        batch_shape: torch.Size = torch.Size(),
    ) -> None:
        """
        Args:
            graph: Adjacency Matrix for the underlying SEM
            node_names: The names of each node in the graph
            event_shape: Shape of a sample from the distribution
            batch_shape: Batch shape of the distribution
        """
        self._graph = graph
        self.node_names = node_names
        assert node_names, "Can't have an empty SEM"
        assert graph.shape[-2:] == (
            len(node_names),
            len(node_names),
        ), "Graph adjacency matrix must have shape [num_nodes, num_nodes] (excluding batch dimensions)"
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
        """Return the SEM associated with the interventions"""

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
    effects: Optional[list[str]] = None,
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
    # TODO: do we want to average over multiple "sample_to_noise" values for the discrete variables
    # where this is not a 1:1 mapping?
    return sem.do(interventions=intervention).noise_to_sample(sem.sample_to_noise(factual_data))


def ate(
    sem: SEM,
    intervention_a: TensorDict,
    intervention_b: TensorDict,
    effects: Optional[list[str]] = None,
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
