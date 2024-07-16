from typing import Optional

import torch
from tensordict import TensorDict

from causica.datasets.tensordict_utils import expand_td_with_batch_shape
from causica.distributions import JointNoiseModule
from causica.functional_relationships import (
    DoFunctionalRelationships,
    FunctionalRelationships,
    TemporalEmbedFunctionalRelationships,
    create_do_functional_relationship,
)
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
        log_func_rescale: Optional[FunctionalRelationships] = None,
    ):
        """
        Args:
            graph: Adjacency Matrix
            noise_dist: A dictionary of callables for generating distributions from predictions
            func: The functional relationship that can map from the tensor representation to the to itself.
            log_func_rescale: The log of the functional relationship that applies a data-dependent operation on the noise.
        """

        self.batch_shape_f = func.batch_shape
        self.batch_shape_g = torch.Size(graph.shape[:-2])

        if isinstance(func, TemporalEmbedFunctionalRelationships):
            raise ValueError(f"Functional relationship {func} is a TemporalEmbedFunctionalRelationships.")

        if len(self.batch_shape_f) == 0 and len(self.batch_shape_g) > 0:
            self.batch_shape_f = torch.Size((1,))
            func.batch_shape = self.batch_shape_f

        super().__init__(
            graph=graph,
            node_names=noise_dist.keys(),
            batch_shape=torch.Size(self.batch_shape_f + self.batch_shape_g),
        )
        self.noise_dist = noise_dist
        self.func = func
        self.log_func_rescale = log_func_rescale

    def log_prob(self, value: TensorDict) -> torch.Tensor:
        """
        Compute the log prob of the observations

        Args:
            value: a dictionary of `sample_shape + batch_shape + [variable_dim]` shaped tensors
        Return:
            A tensor of shape `sample_shape + batch_shape` of the log probability of each event
        """
        expanded_value = expand_td_with_batch_shape(value, self.batch_shape)
        if self.log_func_rescale is None:
            res = self.noise_dist(self.func(expanded_value, self.graph)).log_prob(expanded_value)
        else:
            res = self.noise_dist(
                (self.func(expanded_value, self.graph), self.log_func_rescale(expanded_value, self.graph))
            ).log_prob(expanded_value)
        return res

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
        x = expand_td_with_batch_shape(noise.clone(), self.batch_shape).zero_()
        expanded_noise = expand_td_with_batch_shape(noise, self.batch_shape)
        for _ in self.node_names:
            if self.log_func_rescale is None:
                x.update_(self.noise_dist(self.func(x, self.graph)).noise_to_sample(expanded_noise))
            else:
                x.update_(
                    self.noise_dist((self.func(x, self.graph), self.log_func_rescale(x, self.graph))).noise_to_sample(
                        expanded_noise
                    )
                )
        if isinstance(self.func, DoFunctionalRelationships):
            do = self.func.do
            if do.batch_dims > 0:
                do = do[(...,) + (None,) * len(self.batch_shape_g)]
            expanded_do = do.expand(*x.batch_size)
            x.update(expanded_do, inplace=True)
            x = self.func.func.tensor_to_td.order_td(x)
        return x

    def sample_to_noise(self, sample: TensorDict) -> TensorDict:
        """
        For a given sample get the noise vector.

        Args:
            sample: samples from the sem each of shape `sample_shape + batch_shape + event_shape`
        Return:
            Dictionary of tensors representing the noise shape `sample_shape + batch_shape + noise_shape`
        """
        expanded_sample = expand_td_with_batch_shape(sample, self.batch_shape)
        if self.log_func_rescale is None:
            res = self.noise_dist(self.func(expanded_sample, self.graph)).sample_to_noise(expanded_sample)
        else:
            res = self.noise_dist(
                (self.func(expanded_sample, self.graph), self.log_func_rescale(expanded_sample, self.graph))
            ).sample_to_noise(sample)

        return res

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

        return self.noise_dist().sample(torch.Size(torch.Size(sample_shape) + self.batch_shape))

    def do(self, interventions: TensorDict) -> "DistributionParametersSEM":
        """Return the SEM associated with the interventions.

        If the interventions have a batch shape, then the returned functions will have the same batch shape to batch
        the computation across interventions.
        """
        do_log_func = None
        if isinstance(self.func, DoFunctionalRelationships):
            if self.log_func_rescale is not None:
                if not isinstance(self.log_func_rescale, DoFunctionalRelationships):
                    raise ValueError(
                        "log_func_rescale must be DoFunctionalRelationships when func is DoFunctionalRelationships"
                    )

            if set(interventions.keys()) & set(self.func.do.keys()):
                raise ValueError("Cannot intervene on already intervened nodes")
            prev_interventions = self.func.do
            if interventions.batch_dims == 0 and prev_interventions.batch_dims > 0:
                interventions = interventions.expand(prev_interventions.batch_size)
            if prev_interventions.batch_dims == 0 and interventions.batch_dims > 0:
                prev_interventions = prev_interventions.expand(interventions.batch_size)
            if interventions.batch_size != prev_interventions.batch_size:
                raise ValueError(
                    f"Interventions batch shape {interventions.batch_size} does not match do batch shape {self.func.do.batch_size}"
                )
            interventions_updated = interventions.update(prev_interventions)
            graph_updated = self.func.pad_intervened_graphs(self.graph)
            do_func, do_graphs = create_do_functional_relationship(
                interventions=interventions_updated,
                func=self.func.func,
                graph=graph_updated,
            )

            if self.log_func_rescale is not None:
                interventions_replaced = interventions_updated.apply(torch.zeros_like)
                do_log_func, _ = create_do_functional_relationship(
                    interventions=interventions_replaced,
                    func=self.log_func_rescale.func,
                    graph=graph_updated,
                )

        else:

            do_func, do_graphs = create_do_functional_relationship(
                interventions=interventions, func=self.func, graph=self.graph
            )

            if self.log_func_rescale is not None:
                interventions_replaced = interventions.apply(torch.zeros_like)
                do_log_func, _ = create_do_functional_relationship(
                    interventions=interventions_replaced, func=self.log_func_rescale, graph=self.graph
                )

        intervened_node_names = set(interventions.keys())
        unintervened_node_names = [name for name in self.node_names if name not in intervened_node_names]  # keep order
        return DistributionParametersSEM(
            graph=do_graphs,
            noise_dist=self.noise_dist[unintervened_node_names],
            func=do_func,
            log_func_rescale=do_log_func,
        )
