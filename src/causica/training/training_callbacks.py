from typing import Deque

import numpy as np


class AverageMetricTracker:
    """A class to keep the smallest value of the rolling average of a metric over time.

    Args:
        averaging_period: Number of steps to average over.
    """

    def __init__(self, averaging_period: int = 10):
        self._averaging_period = averaging_period
        self.min_value = np.inf
        self.queue: Deque = Deque([], maxlen=self._averaging_period)
        self.rolling_sum = 0.0

    @property
    def average(self):
        return self.rolling_sum / len(self.queue)

    def step(self, value: float) -> bool:
        """Add a new value to the tracker."""
        removed_value = self.queue.popleft() if len(self.queue) == self._averaging_period else 0.0
        self.queue.append(value)
        self.rolling_sum += value - removed_value

        if (current_average := self.average) < self.min_value:
            self.min_value = current_average
            return True
        return False

    def reset(self):
        """Reset the tracker."""
        self.min_value = np.inf
        self.queue = Deque([], maxlen=self._averaging_period)
        self.rolling_sum = 0.0
