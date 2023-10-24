from typing import Iterable, Optional, Union

from torchmetrics import Metric, MetricCollection
from torchmetrics.wrappers import MultitaskWrapper


def create_metrics_for_variables(variables: Iterable[str], metrics: MetricCollection) -> MultitaskWrapper:
    """
    This function will create a MultiTaskWrapper obj with keys to be the variable names.

    Args:
        variables: Variables name list to calculate the metrics for.
        metrics: MetricCollection to store the metrics we want to compute.

    Returns:
        MultitaskWrapper obj with keys to be the variable names.
    """

    metrics_dict: dict[str, Union[Metric, MetricCollection]] = {
        key: metrics.clone(postfix=f".{key}") for key in variables
    }

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
