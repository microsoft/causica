from typing import List, NamedTuple, Optional

import numpy as np


class InterventionMetadata(NamedTuple):

    columns_to_nodes: List[int]

    def to_dict(self):
        return self._asdict()


class InterventionData(NamedTuple):
    """Class that acts as a container for observational (rank-1), interventional (rank-2) or counterfactual (rank-3) data.

    This data object can be serialized by converting to a dict, taking the form

                {
                        "intervention_idxs": Optional[np.ndarray]
                        "intervention_values": Optional[np.ndarray]
                        "test_data": np.ndarray
                        "conditioning_idxs": Optional[np.ndarray] = None
                        "conditioning_values": Optional[np.ndarray] = None
                        "effect_idxs": Optional[np.ndarray] = None
                        "intervention_reference": Optional[np.ndarray] = None
                        "reference_data": Optional[np.ndarray] = None
                },

    Args:
        conditioning_idxs: np.ndarray. 1d array containing the indices of each variable on which we condition on. For counterfactuals,
            all variables should be conditioned on.
        conditioning_values: np.ndarray. 1d array containing the values being assigned to the conditioned variables.
        effect_idxs: np.ndarray. 1d array containing the indices of each variable for which we want to evaluate the effect of the treatment.
        intervention_idxs: np.ndarray. 1d array containing the indices of each variable on which an intervention is made.
        intervention_values: np.ndarray. 1d array containing the values being assigned to the intervened variables.
        intervention_reference: np.ndarray 1d array containing reference values for interventions.
        test_data: np.ndarray. Samples from intervened distribution.
        reference_data: np.ndarray. Samples from intervened distribution with reference intervention.
    """

    intervention_idxs: Optional[np.ndarray]
    intervention_values: Optional[np.ndarray]
    test_data: np.ndarray
    conditioning_idxs: Optional[np.ndarray] = None
    conditioning_values: Optional[np.ndarray] = None
    effect_idxs: Optional[np.ndarray] = None
    intervention_reference: Optional[np.ndarray] = None
    reference_data: Optional[np.ndarray] = None

    def to_dict(self):
        # When converting to dict, numpy arrays are converted to lists
        result = self._asdict()
        for k, v in result.items():
            if v is not None:
                result[k] = v.tolist()
        return result

    @classmethod
    def from_dict(cls, input_dict):
        type_converted_input = {k: np.atleast_1d(v) if v is not None else None for k, v in input_dict.items()}
        return cls(**type_converted_input)


class InterventionDataContainer(NamedTuple):
    """A container object for data from multiple interventional environments.

    This object can be serialized and has the following form

        {
        "metadata": {
            "columns_to_nodes": List[int]
        }
        "environments": [
            {
                "intervention_idxs": Optional[np.ndarray]
                "intervention_values": Optional[np.ndarray]
                "test_data": np.ndarray
                "conditioning_idxs": Optional[np.ndarray] = None
                "conditioning_values": Optional[np.ndarray] = None
                "effect_idxs": Optional[np.ndarray] = None
                "intervention_reference": Optional[np.ndarray] = None
                "reference_data": Optional[np.ndarray] = None
            },
            ...
        ]

    Args:
        metadata: InterventionMetadata. Contains meta-information about the SEM.
        environments: List[InterventionData]. Contains data from different interventional environments.

    """

    metadata: InterventionMetadata
    environments: List[InterventionData]

    def to_dict(self):
        result = self._asdict()
        for k, v in result.items():
            if isinstance(v, list):
                result[k] = [x.to_dict() for x in v]
            else:
                result[k] = v.to_dict()
        return result

    @classmethod
    def from_dict(cls, input_dict):
        assert set(input_dict.keys()) == {"metadata", "environments"}
        input_dict["metadata"] = InterventionMetadata(**input_dict["metadata"])
        input_dict["environments"] = [InterventionData.from_dict(data_dict) for data_dict in input_dict["environments"]]
        return cls(**input_dict)

    def validate(self, counterfactual=False):

        if counterfactual:
            node_set = set(self.metadata.columns_to_nodes)
            for environment in self.environments:
                # For counterfactuals, validate that conditioning is on every node
                conditioning_node_set = set(environment.conditioning_idxs)
                assert node_set == conditioning_node_set, "Counterfactual data expects conditioning on every node"
                # And validate that values and intervention sample shapes match
                assert (
                    environment.conditioning_values.shape[0] == environment.test_data.shape[0]
                ), "Counterfactual data expects the conditioning to be of equivalent shape to the interventional data."
