import torch
from tensordict import TensorDict

from causica.distributions import JointNoiseModule
from causica.functional_relationships import FunctionalRelationships, create_do_functional_relationship
from causica.sem.structural_equation_model import SEM


class DistributionParametersSEM(SEM):
    """
    A Structural Equation Model where the functional forward pass generates the parameters of the distributions to be sampled.

    This is more general than the simple additive case.
    """

    arg_constraints: dict = {}

    def __init__(
        self,
        graph: torch.Tensor,
        noise_dist: JointNoiseModule,
        func: FunctionalRelationships,
    ):
        """
        Args:
            graph: Adjacency Matrix
            noise_dist: A dictionary of callables for generating distributions from predictions
            func: The functional relationship that can map from the tensor representation to the to itself.
        """
        super().__init__(graph=graph, node_names=noise_dist.keys())
        self.noise_dist = noise_dist
        self.func = func

    def log_prob(self, value: TensorDict) -> torch.Tensor:
        """
        Compute the log prob of the observations

        Args:
            value: a dictionary of `sample_shape + batch_shape + [variable_dim]` shaped tensors
        Return:
            A tensor of shape `sample_shape + batch_shape` of the log probability of each event
        """
        return self.noise_dist(self.func(value, self.graph)).log_prob(value)

    def noise_to_sample(self, noise: TensorDict) -> TensorDict:
        """
        For a given noise vector, return the corresponding sample values.

        Args:
            noise: Dictionary of tensors representing the noise shape
            `sample_shape + batch_shape + noise_shape`
        Return:
            Dictionary of samples from the sem corresponding to the noise, shape
            `sample_shape + batch_shape + event_shape`
        """
        x = noise.clone().zero_()
        for _ in self.node_names:
            x.update_(self.noise_dist(self.func(x, self.graph)).noise_to_sample(noise))
        return x

    def sample_to_noise(self, sample: TensorDict) -> TensorDict:
        """
        For a given sample get the noise vector.

        Args:
            sample: samples from the sem each of shape `sample_shape + batch_shape + event_shape`
        Return:
            Dictionary of tensors representing the noise shape `sample_shape + batch_shape + noise_shape`
        """
        return self.noise_dist(self.func(sample, self.graph)).sample_to_noise(sample)

    @torch.no_grad()
    def sample_noise(self, sample_shape: torch.Size = torch.Size()) -> TensorDict:
        """
        Sample the noise vector for the distribution.

        Args:
            sample_shape: shape of the returned noise samples
        Return:
            Dictionary of samples from the noise distribution for each node, shape
            `sample_shape + batch_shape + event_shape`.
        """
        return self.noise_dist().sample(sample_shape)

    def do(self, interventions: TensorDict) -> "DistributionParametersSEM":
        """Return the SEM associated with the interventions"""
        do_func, do_graphs = create_do_functional_relationship(
            interventions=interventions, func=self.func, graph=self.graph
        )
        intervened_node_names = set(interventions.keys())
        unintervened_node_names = [name for name in self.node_names if name not in intervened_node_names]  # keep order
        return DistributionParametersSEM(
            graph=do_graphs,
            noise_dist=self.noise_dist[unintervened_node_names],
            func=do_func,
        )
