from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class IMetricsLogger(ABC):
    @abstractmethod
    def log_value(self, metric_name: str, value: Any, log_to_parent: Optional[bool] = False):
        raise NotImplementedError()

    @abstractmethod
    def log_list(self, metric_name: str, values: List[Any], log_to_parent: Optional[bool] = False):
        raise NotImplementedError()

    @abstractmethod
    def set_tags(self, tags: Dict[str, Any], log_to_parent: Optional[bool] = False):
        raise NotImplementedError()

    @abstractmethod
    def finalize(self):
        raise NotImplementedError()

    @abstractmethod
    def log_dict(self, metrics: Dict[str, Any], log_to_parent: Optional[bool] = False):
        """Log a dictionary with metrics. The metics dict

        Args:
            metrics (str): metrics dict. It can be nested. If so, the keys will be flattened: {'a': {'b': 2}} will be logged as {'a.b': 2}
            log_to_parent (bool): Specify whether the metric should be also logged to parent run. Defaults to False
        """
        raise NotImplementedError()


class ISystemMetricsLogger(ABC):
    @abstractmethod
    def start_log(self):
        raise NotImplementedError()

    @abstractmethod
    def end_log(self):
        raise NotImplementedError()
