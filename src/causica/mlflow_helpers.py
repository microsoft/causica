import time
from queue import Queue

import mlflow


class MLFlowBatch:
    """
    A class for collecting MLFlow logs and submitting them as a batch to speed up logging.

    It will automatically batch all logs, once the batch size is reached.

    Accumulated metrics can be manually logged by calling flush.
    """

    def __init__(self, batch_size: int) -> None:
        self._queue: Queue = Queue(maxsize=batch_size)  # mypy requires explicit type annotation here.
        self._client = mlflow.tracking.MlflowClient()
        active_run = mlflow.active_run()
        if active_run is None:
            raise ValueError("No active ML flow run.")
        self._run_id = active_run.info.run_id

    def log_metric(self, key: str, value: float, step: int = 0):
        """Stores a metric in the log batch and pushes if the batch is full."""
        metric = mlflow.entities.Metric(key=key, value=float(value), timestamp=int(time.time()), step=step)
        self._queue.put(metric)
        if self._queue.full():
            self.flush()

    def flush(self):
        """Manually push the accumulated logs."""
        metrics = []
        while not self._queue.empty():
            metrics.append(self._queue.get())
        self._client.log_batch(run_id=self._run_id, metrics=metrics)
