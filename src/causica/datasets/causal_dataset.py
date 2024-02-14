from dataclasses import dataclass, field

import torch
from tensordict import TensorDictBase

from causica.datasets.causica_dataset_format import CounterfactualWithEffects, InterventionWithEffects


@dataclass
class CausalDataset:
    """Dataclass for storing a causal dataset.

    Args:
        observations: TensorDictBase containing the observations.
        graph: Adjacency matrix of the causal graph.
        noise: Optional, TensorDictBase containing the noise used for generating the observations.
        counterfactuals: Optional, List of CounterfactualWithEffects.
        interventions: Optional, List of InterventionWithEffects.
    """

    observations: TensorDictBase
    graph: torch.Tensor
    node_names: list[str] = field(init=False)  # the nodes that are neither conditioned nor sampled
    noise: TensorDictBase | None = None
    counterfactuals: list[CounterfactualWithEffects] | None = None
    interventions: list[InterventionWithEffects] | None = None

    def __post_init__(self):
        if len(self.observations.batch_size) != 1:
            raise ValueError(f"Batch size must be a scalar, got {self.observations.batch_size}")
        if self.graph.ndim != 2:
            raise ValueError("Expected graph dimension to be 2")

        if self.graph.shape[-1] != self.graph.shape[-2]:
            raise ValueError("The graph must be a square matrix.")

        if self.graph.shape[-1] != len(list(self.observations.keys())):
            raise ValueError(
                f"Graph shape {self.graph.shape} does not match observations: {list(self.observations.keys())}"
            )

        if self.noise is not None and self.noise.batch_size != self.observations.batch_size:
            raise ValueError("Noise batch size does not match observations batch size")

        if self.noise is not None and set(self.noise.keys()) != set(self.observations.keys()):
            raise ValueError("Noise keys do not match observations keys")

        if self.counterfactuals is not None and not all(
            set(cf[0].factual_data.keys()) == set(self.observations.keys()) for cf in self.counterfactuals
        ):
            raise ValueError("Some counterfactual factual data keys do not match observations keys")

        if self.interventions is not None and not all(
            set(intervention[0].intervention_data.keys()) == set(self.observations.keys())
            for intervention in self.interventions
        ):
            raise ValueError("Some intervention data keys do not match observations keys")

        self.node_names = list(self.observations.keys())
