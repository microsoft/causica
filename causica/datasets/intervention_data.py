from typing import NamedTuple, Optional

import numpy as np


class InterventionData(NamedTuple):
    """Class that acts as a container for interventional (rank-2) or counterfactual (rank-3) data.

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

    intervention_idxs: np.ndarray
    intervention_values: Optional[np.ndarray]
    test_data: np.ndarray
    conditioning_idxs: Optional[np.ndarray] = None
    conditioning_values: Optional[np.ndarray] = None
    effect_idxs: Optional[np.ndarray] = None
    intervention_reference: Optional[np.ndarray] = None
    reference_data: Optional[np.ndarray] = None
