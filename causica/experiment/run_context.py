from typing import Any, Callable, Dict, List, Optional

from .imetrics_logger import IMetricsLogger, ISystemMetricsLogger


def mock_download_dataset(dataset_name: str, data_dir: str):
    raise NotImplementedError("No download_dataset functionality provided")


def aml_step(func: Callable, _: bool) -> Callable:
    return func


class MockMetricLogger(IMetricsLogger):
    def log_value(self, metric_name: str, value: Any, log_to_parent: Optional[bool] = False):
        pass

    def log_list(self, metric_name: str, values: List[Any], log_to_parent: Optional[bool] = False):
        pass

    def set_tags(self, tags: Dict[str, Any], log_to_parent: Optional[bool] = False):
        pass

    def finalize(self):
        pass

    def log_dict(self, metrics: Dict[str, Any], log_to_parent: Optional[bool] = False):
        pass


class MockSystemMetricsLogger(ISystemMetricsLogger):
    def start_log(self):
        pass

    def end_log(self):
        return {}


class RunContext:
    """
    Run context which carries information about the computation environment

    States whether the computation is running in aml, how to download dataset on machine etc.
    """

    def __init__(self):
        # Function for downloading the dataset
        self.download_dataset = mock_download_dataset
        # Function for saying whether it is aml run or not
        self.is_azureml_run = lambda: False
        # Metrics logger used for a run
        self.mock_metrics_logger = MockMetricLogger()
        self.metrics_logger = self.mock_metrics_logger
        # System metrics logger used for a run
        self.sys_mock_metric_logger = MockSystemMetricsLogger()  # type:ISystemMetricsLogger
        self.system_metrics_logger = self.sys_mock_metric_logger
        # Evaluation pipeline used for a run
        # If no evaluation pipeline used, None is passed
        self.pipeline = None  # type:ignore
        # Decorator function for methods which are supposed to
        # be used as separate AML step:
        # these methods can be either in creation or running mode:
        # (which is second argument to the decorator)
        # a) in creation mode, their input will be serialized, so
        # it can be transferred to the cloud, where it can be run
        # b) in running mode, the method is simply run
        self.aml_step = aml_step
