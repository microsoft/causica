import torch
import torch.distributions as td
from torch.distributions.constraints import Constraint

from causica.distributions.adjacency import AdjacencyDistribution
from causica.distributions.distribution_module import DistributionModule
from causica.distributions.noise.joint import JointNoiseModule
from causica.functional_relationships import FunctionalRelationships, TemporalEmbedFunctionalRelationships
from causica.sem.distribution_parameters_sem import DistributionParametersSEM
from causica.sem.temporal_distribution_parameters_sem import TemporalDistributionParametersSEM


class SEMDistribution(td.Distribution):
    """A distribution over structural equation models.

    Samples are instances of DistributionParametersSEM. Note however that this was created before
    pytorch set the expected type of samples to torch.Tensor, so this is breaking the types a bit.

    The distribution is essentially the same as the given adjacency distribution but with samples converted to SEMs.
    Therefore, all distribution properties such as entropy, mean and mode are given by the equivalent properties for the
    adjacency distribution.
    """

    arg_constraints: dict[str, Constraint] = {}

    def __init__(
        self,
        adjacency_dist: AdjacencyDistribution,
        noise_module: JointNoiseModule,
        functional_relationships: FunctionalRelationships,
    ):
        """
        Args:
            adjacency_dist: Distribution from which adjacency matrices are sampled to construct SEMs.
            noise_module: The noise module for any SEM of this distribution.
            functional_relationships: The functional relationship for any SEM of this distribution.
        """
        super().__init__()
        self._adjacency_dist = adjacency_dist
        self._noise_module = noise_module
        self._functional_relationships = functional_relationships

    def _create_sems(self, graphs: torch.Tensor) -> list[DistributionParametersSEM]:
        graphs = graphs.reshape(-1, *self._adjacency_dist.event_shape)
        return [
            DistributionParametersSEM(
                graph=graph,
                noise_dist=self._noise_module,
                func=self._functional_relationships,
            )
            for graph in graphs.unbind(dim=0)
        ]

    def sample(self, sample_shape: torch.Size = torch.Size()):
        graphs = self._adjacency_dist.sample(sample_shape)
        if not sample_shape:
            graphs = graphs[None, ...]
        return self._create_sems(graphs)

    def relaxed_sample(self, sample_shape: torch.Size = torch.Size(), temperature: float = 0.0):
        graphs = self._adjacency_dist.relaxed_sample(sample_shape=sample_shape, temperature=temperature)
        return self._create_sems(graphs)

    def entropy(self) -> torch.Tensor:
        return self._adjacency_dist.entropy()

    @property
    def mean(self) -> DistributionParametersSEM:  # type: ignore
        return self._create_sems(self._adjacency_dist.mean)[0]

    @property
    def mode(self) -> DistributionParametersSEM:  # type: ignore
        return self._create_sems(self._adjacency_dist.mode)[0]

    def log_prob(self, value: DistributionParametersSEM) -> torch.Tensor:  # type: ignore
        return self._adjacency_dist.log_prob(value.graph)


class SEMDistributionModule(DistributionModule[SEMDistribution]):
    """Represents a SEMDistribution with learnable parameters."""

    def __init__(
        self,
        adjacency_module: DistributionModule[AdjacencyDistribution],
        functional_relationships: FunctionalRelationships,
        noise_module: JointNoiseModule,
    ):
        super().__init__()
        self.adjacency_module = adjacency_module
        self.functional_relationships = functional_relationships
        self.noise_module = noise_module

    def forward(self) -> SEMDistribution:
        return SEMDistribution(
            adjacency_dist=self.adjacency_module(),
            noise_module=self.noise_module,
            functional_relationships=self.functional_relationships,
        )


class TemporalSEMDistribution(SEMDistribution):
    def __init__(
        self,
        adjacency_dist: AdjacencyDistribution,
        noise_module: JointNoiseModule,
        functional_relationships: TemporalEmbedFunctionalRelationships,
    ):
        super().__init__(adjacency_dist, noise_module, functional_relationships)
        self._functional_relationships: TemporalEmbedFunctionalRelationships = functional_relationships

    def _create_sems(self, graphs: torch.Tensor) -> list[TemporalDistributionParametersSEM]:  # type: ignore
        graphs = graphs.reshape(-1, *self._adjacency_dist.event_shape)
        return [
            TemporalDistributionParametersSEM(
                graph=graph,
                noise_dist=self._noise_module,
                func=self._functional_relationships,
            )
            for graph in graphs.unbind(dim=0)
        ]


class TemporalSEMDistributionModule(DistributionModule[TemporalSEMDistribution]):
    """Represents a SEMDistribution with learnable parameters."""

    def __init__(
        self,
        adjacency_module: DistributionModule[AdjacencyDistribution],
        functional_relationships: TemporalEmbedFunctionalRelationships,
        noise_module: JointNoiseModule,
    ):
        super().__init__()
        self.adjacency_module = adjacency_module
        self.functional_relationships = functional_relationships
        self.noise_module = noise_module

    def forward(self) -> TemporalSEMDistribution:
        return TemporalSEMDistribution(
            adjacency_dist=self.adjacency_module(),
            noise_module=self.noise_module,
            functional_relationships=self.functional_relationships,
        )
