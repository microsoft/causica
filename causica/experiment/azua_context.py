from typing import Any, Callable, Dict, List, Optional

from dependency_injector import containers, providers

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


# Azua context which carries the information on computation
# environment (e.g. whether running in aml, how to download dataset on machine etc.)
# This is injected by using dependency injection throhout experiment code
# (i.e. causica/run_experiment.py script)


class AzuaContext(containers.DeclarativeContainer):

    # Function for downloading the dataset
    download_dataset = providers.Callable(mock_download_dataset)
    # Function for saying whether it is aml run or not
    is_azureml_run = providers.Callable(lambda: False)
    # Metrics logger used for a run
    mock_metrics_logger = MockMetricLogger()  # type:IMetricsLogger
    metrics_logger = providers.Object(mock_metrics_logger)
    # System metrics logger used for a run
    sys_mock_metric_logger = MockSystemMetricsLogger()  # type:ISystemMetricsLogger
    system_metrics_logger = providers.Object(sys_mock_metric_logger)
    # Evaluation pipeline used for a run
    # If no evaluation pipeline used, None is passed
    pipeline = providers.Object(None)  # type:ignore # TODO: Add typing by adding IEvaluationPipeline to azua/?
    # Decorator function for methods which are supposed to
    # be used as separate AML step:
    # these methods can be either in creation or running mode:
    # (which is secon argument to the decorator)
    # a) in creation mode, their input will be serialized, so
    # it can be transferred to the cloud, where it can be run
    # b) in running mode, the method is simply run
    aml_step = providers.Callable(aml_step)
