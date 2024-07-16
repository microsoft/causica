import torch
from tensordict import TensorDict, TensorDictBase

from causica.datasets.tensordict_utils import expand_td_with_batch_shape
from causica.distributions import JointNoiseModule
from causica.functional_relationships import DoFunctionalRelationships, TemporalEmbedFunctionalRelationships
from causica.sem.structural_equation_model import SEM


def split_lagged_and_instanteneous_values(td: TensorDictBase) -> tuple[TensorDict, TensorDict]:
    """Splits a temporal TensorDict into lagged and instantaneous values.

    Args:
        td: The temporal TensorDict with shape (..., context_length, (variable_shape)). The batch size of the TensorDict
            should not include the context length, ie. td.batch_size = (...) and not td.batch_size = (..., context_length).

    Returns:
        A tuple with the lagged values and the instantaneous values.
    """
    orig_batch_size = td.batch_size
    new_batch_size = next(td.values()).shape[:-1]
    td.batch_size = new_batch_size

    lagged, instantaneous = td[..., :-1], td[..., -1]

    td.batch_size = orig_batch_size
    lagged.batch_size = orig_batch_size
    instantaneous.batch_size = orig_batch_size

    return lagged, instantaneous


def concatenate_lagged_and_instaneous_values(
    lagged_values: TensorDictBase,
    instantaneous_values: TensorDictBase,
) -> TensorDict:
    """Concatenate the lagged and instantaneous values.

    Args:
        instantaneous_values: The TensorDict with the instantaneous values with shape (..., (variable_shape))
        lagged_values: The TensorDict with the lagged values with shape (..., context_length, (variable_shape))

    Returns:
        A new TensorDict with the concatenated lagged and instantaneous values.
    """
    return TensorDict(
        {
            k: torch.cat([lagged_values[k], v[..., None, :]], dim=-2)
            for k, v in instantaneous_values.items(include_nested=True, leaves_only=True)
        },
        batch_size=instantaneous_values.batch_size,
    )


class TemporalDistributionParametersSEM(SEM):
    """
    A Structural Equation Model where the functional forward pass generates the parameters of the distributions to be sampled.

    This is more general than the simple additive case.
    """

    def __init__(
        self,
        graph: torch.Tensor,
        noise_dist: JointNoiseModule,
        func: TemporalEmbedFunctionalRelationships,
    ):
        """
        Args:
            graph: Adjacency Matrix
            noise_dist: A dictionary of callables for generating distributions from predictions
            func: The functional relationship that can map from the tensor representation to the to itself.
        """

        self.batch_shape_f = func.batch_shape
        batch_shape_g = torch.Size(graph.shape[:-3])
        self.context_length = graph.shape[-3]

        if self.context_length != func.nn.context_length:
            raise ValueError(
                f"Graph context length {self.context_length} does not match functional relationship context length {func.context_length}"
            )

        if len(self.batch_shape_f) == 0 and len(batch_shape_g) > 0:
            self.batch_shape_f = torch.Size((1,))
            func.batch_shape = self.batch_shape_f

        super().__init__(
            graph=graph,
            node_names=noise_dist.keys(),
            batch_shape=torch.Size(self.batch_shape_f + batch_shape_g),
        )
        self.noise_dist = noise_dist
        self.func = func

    def log_prob(self, value: TensorDict) -> torch.Tensor:
        """
        Compute the log prob of the observations

        Args:
            value: a dictionary of `sample_shape + batch_shape + context_length + [variable_dim]` shaped tensors
        Return:
            A tensor of shape `sample_shape + batch_shape` of the log probability of each event
        """
        if value[value.sorted_keys[0]].shape[-2] != self.context_length:
            raise ValueError(
                f"Value shape {value[value.sorted_keys[0]].shape} does not match graph shape {self.graph.shape}."
                " The last batch dimension of the value should match the context length."
            )
        expanded_value = expand_td_with_batch_shape(value, self.batch_shape)
        _lagged, test_value = split_lagged_and_instanteneous_values(expanded_value)  # pylint: disable=unused-variable
        return self.noise_dist(self.func(expanded_value, self.graph)).log_prob(test_value)

    def noise_to_sample(self, noise: TensorDict) -> TensorDict:
        """
        For a given noise vector, return the corresponding sample values.

        Args:
            noise: Dictionary of tensors representing the noise shape
            `sample_shape + batch_shape + context_length + noise_shape`, where the context length dimension contains
            the observed history as well as the noise for the current time step.
        Return:
            Dictionary of samples from the sem corresponding to the noise, shape
            `sample_shape + batch_shape + event_shape`
        """
        x = expand_td_with_batch_shape(noise.clone(), self.batch_shape).zero_()
        history, noise = split_lagged_and_instanteneous_values(noise)
        expanded_noise = expand_td_with_batch_shape(noise, self.batch_shape)
        for _ in self.node_names:
            new_sample = self.noise_dist(self.func(x, self.graph)).noise_to_sample(expanded_noise)
            x.update_(concatenate_lagged_and_instaneous_values(history, new_sample))
        x = split_lagged_and_instanteneous_values(x)[-1]
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
            sample: samples from the sem each of shape `sample_shape + batch_shape + context_length + event_shape`
        Return:
            Dictionary of tensors representing the noise shape `sample_shape + batch_shape + noise_shape`
        """
        if sample[sample.sorted_keys[0]].shape[-2] != self.context_length:
            raise ValueError(
                f"Value shape {sample[sample.sorted_keys[0]].shape} does not match graph shape {self.graph.shape}."
                " The last batch dimension of the value should match the context length."
            )

        expanded_sample = expand_td_with_batch_shape(sample, self.batch_shape)
        _lagged, test_sample = split_lagged_and_instanteneous_values(expanded_sample)  # pylint: disable=unused-variable
        return self.noise_dist(self.func(expanded_sample, self.graph)).sample_to_noise(test_sample)

    def do(self, interventions: TensorDict) -> "TemporalDistributionParametersSEM":
        """Return the SEM associated with the interventions.

        If the interventions have a batch shape, then the returned functions will have the same batch shape to batch
        the computation across interventions.
        """
        _ = interventions
        raise NotImplementedError("Interventions are not yet supported for temporal graphs")

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
