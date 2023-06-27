from typing import Sequence

from lightning_utilities.core.rank_zero import rank_zero_only
from mlflow.entities import Metric, Param, RunTag
from pytorch_lightning.loggers import MLFlowLogger


class BufferingMlFlowLogger(MLFlowLogger):
    """MlFlowLogger that buffers metrics on logging and flushes on finalize or when the buffer is full."""

    def __init__(self, buffer_size: int, *args, **kwargs):
        """
        Args:
            buffer_size: The maximum number of metrics to buffer before flushing
            *args: Passed to `MLFlowLogger`
            **kwargs: Passed to `MLFlowLogger`
        """
        super().__init__(*args, **kwargs)
        self._buffer_size = buffer_size
        self._buffer: list[Metric] = []
        self._original_log_batch = self.experiment.log_batch
        self.experiment.log_batch = self._buffer_log_batch_metrics(self.experiment.log_batch)

    @rank_zero_only
    def _buffer_log_batch_metrics(self, original_log_batch):
        """Returns a decorated `log_batch` that buffers metrics and flushes them when the buffer is full."""

        def log_batch(
            run_id: str,
            metrics: Sequence[Metric] = (),
            params: Sequence[Param] = (),
            tags: Sequence[RunTag] = (),
        ) -> None:
            if metrics:
                self._buffer.extend(metrics)
                if len(self._buffer) >= self._buffer_size:
                    self.flush()
            if params or tags:
                original_log_batch(run_id=run_id, params=params, tags=tags)

        return log_batch

    def get_buffer_count(self) -> int:
        """Return the current number of buffered messages."""
        return len(self._buffer)

    @rank_zero_only
    def flush(self):
        if self._buffer:
            self._original_log_batch(run_id=self.run_id, metrics=self._buffer)
            self._buffer.clear()

    @rank_zero_only
    def finalize(self, *args, **kwargs) -> None:
        self.flush()
        return super().finalize(*args, **kwargs)

    @rank_zero_only
    def __del__(self) -> None:
        self.flush()
