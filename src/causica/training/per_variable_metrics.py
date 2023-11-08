from typing import Any, Iterable, Optional, Type, Union

from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError
from torchmetrics.wrappers import MultitaskWrapper


class MeanAbsoluteErrorWithThreshold(MeanAbsoluteError):
    """This compute the MAE with a minimum filter on the target values.

    This compute the MAE for target that is higher than the min_threshold.
    """

    def __init__(self, min_threshold: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = min_threshold

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with filtered predictions and targets."""
        # filter based on the threshold
        mask = target > self.threshold
        filtered_preds = preds[mask]
        filtered_target = target[mask]
        super().update(filtered_preds, filtered_target)


class MeanAbsolutePercentageErrorWithThreshold(MeanAbsolutePercentageError):
    """This compute the MAPE with a minimum filter on the target values.

    This compute the MAPE for target that is higher than the min_threshold.
    """

    def __init__(self, min_threshold: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.threshold = min_threshold

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with filtered predictions and targets."""
        # filter based on the threshold
        mask = target > self.threshold
        filtered_preds = preds[mask]
        filtered_target = target[mask]
        super().update(filtered_preds, filtered_target)


def create_metrics_for_variables(
    variables: Iterable[str],
    metrics: MetricCollection,
    min_thresholds: Optional[dict[str, float]] = None,
    threshold_metrics: Optional[dict[str, Type[Metric]]] = None,
) -> MultitaskWrapper:
    """
    This function will create a MultiTaskWrapper obj with keys to be the variable names.

    Args:
        variables: Variables name list to calculate the metrics for.
        metrics: MetricCollection to store the metrics we want to compute.
        min_thresholds: A dictionary containing the threshold values for each of the variables.
        threshold_metrics: The dict containing the thresholded torchmetrics class. If None, then no additional threshold
            metrics will be added.

    Returns:
        MultitaskWrapper obj with keys to be the variable names.
    """

    metrics_dict: dict[str, Union[Metric, MetricCollection]] = {
        key: metrics.clone(postfix=f".{key}") for key in variables
    }

    if threshold_metrics is not None and min_thresholds is not None:
        for variable_name, threshold in min_thresholds.items():
            for name, cur_threshold_metric_class in threshold_metrics.items():
                cur_threshold_metric = cur_threshold_metric_class(min_threshold=threshold)
                metrics_dict[variable_name].add_module(name, cur_threshold_metric)

    return MultitaskWrapper(metrics_dict)


def filter_metrics_wrapper(variable_list: Optional[list[str]], metrics_wrapper: MultitaskWrapper) -> MultitaskWrapper:
    """
    This will filter the MultiTaskWrapper obj to select only the variables in variable_list. If None, the metrics will not be filtered.

    Args:
        variable_list: List of variables to filter the metrics_dict.
        metrics_wrapper: MultitaskWrapper obj with keys to be the variable names.

    Returns:
        Filtered MultitaskWrapper obj.
    """
    if variable_list is None:
        return metrics_wrapper

    filtered_dict = {key: metrics_wrapper.task_metrics[key] for key in variable_list}

    return MultitaskWrapper(filtered_dict)
