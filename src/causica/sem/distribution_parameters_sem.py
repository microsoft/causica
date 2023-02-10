import functools
import operator
from typing import Callable, Dict, List

import torch
import torch.distributions as td
from tensordict import TensorDict

from causica.distributions import NoiseAccessibleDistribution
from causica.functional_relationships import FunctionalRelationships, create_do_functional_relationship
from causica.sem.structural_equation_model import SEM


class DistributionParametersSEM(SEM):
    """
    A Structural Equation Model where the functional forward pass generates the parameters of the distributions to be sampled.

    This is more general than the simple additive case.
    """

    arg_constraints: Dict = {}

    def __init__(
        self,
        graph: torch.Tensor,
        node_names: List[str],
        noise_dist: Dict[str, Callable[[torch.Tensor], NoiseAccessibleDistribution]],
        func: FunctionalRelationships,
    ):
        """
        Args:
            graph: Adjacency Matrix
            node_names: The names of each node in the graph
            noise_dist: A dictionary of callables for generating distributions from predictions
            func: The functional relationship that can map from the tensor representation to the to itself.
        """
        super().__init__(graph=graph, node_names=node_names)
        self.noise_dist = noise_dist
        self.func = func
        set_node_names = set(node_names)
        assert set_node_names == noise_dist.keys() == set_node_names

        dist_dict = self._default_init()
        for dist in dist_dict.values():
            assert len(dist.event_shape) == 1, "All node distributions must have vector (rank-1) events"

    def log_prob(self, value: TensorDict) -> torch.Tensor:
        """
        Compute the log prob of the observations

        Args:
            value: a dictionary of sample_shape + batch_shape + [variable_dim] shaped tensors
        Return:
            A tensor of shape sample_shape + batch_shape of the log probability of each event
        """
        dist_params = self.func(value, self.graph)
        cur_log_list = [dist(dist_params[key]).log_prob(value[key]) for key, dist in self.noise_dist.items()]
        return functools.reduce(operator.iadd, cur_log_list)

    def noise_to_sample(self, noise: TensorDict) -> TensorDict:
        """
        For a given noise vector, return the corresponding sample values.

        Args:
            noise: Dictionary of tensors representing the noise shape sample_shape + batch_shape + noise_shape
        Return:
            Dictionary of samples from the sem corresponding to the noise, shape sample_shape + batch_shape + event_shape
        """
        X = noise.clone().zero_()
        for _ in self.node_names:
            forward = self.func(X, self.graph)
            X.update_({key: self.noise_dist[key](val).noise_to_sample(noise[key]) for key, val in forward.items()})
        return X

    def sample_to_noise(self, sample: TensorDict) -> TensorDict:
        """
        For a given sample get the noise vector.

        Args:
            sample: samples from the sem each of shape sample_shape + batch_shape + event_shape
        Return:
            Dictionary of tensors representing the noise shape sample_shape + batch_shape + noise_shape
        """
        forward = self.func(sample, self.graph)
        return TensorDict(
            {key: self.noise_dist[key](val).sample_to_noise(sample[key]) for key, val in forward.items()},
            batch_size=sample.batch_size,
        )

    def sample_noise(self, sample_shape: torch.Size = torch.Size()) -> TensorDict:
        """
        Sample the noise vector for the distribution.

        Args:
            sample_shape: shape of the returned noise samples
        Return:
            Dictionary of samples from the noise distribution for each node, shape sample_shape + batch_shape + event_shape
        """
        dist_dict = self._default_init()
        return TensorDict({key: dist.sample(sample_shape) for key, dist in dist_dict.items()}, batch_size=sample_shape)

    def do(self, interventions: TensorDict) -> "DistributionParametersSEM":
        """Return the SEM associated with the interventions"""
        do_func, do_graphs = create_do_functional_relationship(
            interventions=interventions, func=self.func, graph=self.graph
        )
        unintervened_node_names = [name for name in self.node_names if name not in interventions.keys()]
        intervened_noise_dist = {name: self.noise_dist[name] for name in unintervened_node_names}
        return type(self)(
            graph=do_graphs, node_names=unintervened_node_names, noise_dist=intervened_noise_dist, func=do_func
        )

    def _default_init(self) -> Dict[str, td.Distribution]:
        """
        Initialise the distributions with a default value of zero.

        This is useful for checking properties of the distributions and sampling pure noise

        Using zero to initialise could be a problem for some distributions, but isn't at the minute.
        """
        return {
            key: dist_func(torch.zeros(self.func.variables[key], device=self.graph.device))
            for key, dist_func in self.noise_dist.items()
        }
