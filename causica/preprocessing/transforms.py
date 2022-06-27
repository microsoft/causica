import warnings
from typing import Iterable

import numpy as np
from sklearn.preprocessing import FunctionTransformer

from ..datasets.variables import Variable


class IdentityTransform(FunctionTransformer):
    """Scikit-learn data transformation passing through any data without modification."""

    def __init__(self):
        super().__init__(func=self.identity, inverse_func=self.identity)

    @staticmethod
    def identity(values: np.ndarray) -> np.ndarray:
        """Return values without modification."""
        return values


class UnitScaler(FunctionTransformer):
    """Scikit-learn data transformation for scaling (or squashing) data to the unit hypercube.

    The range of the data is determined by the provided variables.
    """

    def __init__(self, variables: Iterable[Variable]):
        """
        Args:
            variables: Iterable over the variables expected to be transformed
                provided in the same order as data columns.
        """
        # Collect limits for the variables
        lower = []
        upper = []
        for variable in variables:
            lower.append(variable.lower)
            upper.append(variable.upper)

            if variable.lower == variable.upper:
                warnings.warn(
                    f"Variable with name '{variable.name}' has the same upper and lower values. Is this variable a constant?"
                )

        self._lower = np.array(lower)
        self._range: np.ndarray = np.array(upper) - self._lower
        super().__init__(func=self.scale, inverse_func=self.unscale)

    def scale(self, values: np.ndarray) -> np.ndarray:
        """Scale values into the hypercube using pre-determined variable ranges."""
        return (values - self._lower) / self._range

    def unscale(self, scaled_values: np.ndarray) -> np.ndarray:
        """Restore scaled values from the hypercube into the original range."""
        return scaled_values * self._range + self._lower
