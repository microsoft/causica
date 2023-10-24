from dataclasses import dataclass, field

import torch
from tensordict import TensorDictBase


@dataclass
class InterventionData:
    """
    Dataclass to hold the data associated with an intervention

    This represents one intervention and many samples from the intervened distribution

    The class also stores `sampled_nodes`, i.e. the ones that are neither intervened or conditioned on

    Args:
        intervention_data: A `TensorDict` with all the nodes (including intervened/conditioned)
        intervention_values: A dictionary of node names to 1D numpy arrays of the intervened values
        condition_values: A dictionary of node names to 1D numpy arrays of the conditioned values
    """

    intervention_data: TensorDictBase
    intervention_values: TensorDictBase
    condition_values: TensorDictBase
    sampled_nodes: set[str] = field(init=False)  # the nodes that are neither conditioned nor sampled

    def __post_init__(self):
        assert self.intervention_values.batch_size == torch.Size()
        assert self.condition_values.batch_size == torch.Size()

        self.sampled_nodes = (
            set(self.intervention_data.keys())
            - set(self.intervention_values.keys())
            - set(self.condition_values.keys())
        )


@dataclass
class CounterfactualData:
    """
    Dataclass to hold the data associated with a counterfactual

    This represents one intervention and reference and many samples from the intervened and reference
    distributions

    The class also stores `sampled_nodes`, i.e. the ones that are neither intervened or conditioned on

    Args:
        counterfactual_data: A `TensorDict` with all of the node values (including intervened) of counterfactual data
        factual_data: A `TensorDict` with all of the node values of the base observations used for the counterfactuals data.
            This refers to the observations in "What would have happened (CFs) if I would have done (intervention) given
            I observed (base observation).
        intervention_values: A dictionary of node names to 1D numpy arrays of the intervened values
    """

    counterfactual_data: TensorDictBase
    intervention_values: TensorDictBase
    factual_data: TensorDictBase
    sampled_nodes: set[str] = field(init=False)

    def __post_init__(self):
        assert list(self.counterfactual_data.keys()) == list(self.factual_data.keys())
        assert self.counterfactual_data.batch_size == self.factual_data.batch_size
        assert self.intervention_values.batch_size == torch.Size()

        self.sampled_nodes = set(self.counterfactual_data.keys()) - set(self.intervention_values.keys())
