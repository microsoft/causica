import logging
from typing import Any, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
from data_module import ExampleDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from tensordict import TensorDict
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError
from torchmetrics.wrappers import MultitaskWrapper

from causica.datasets.normalization import infer_compatible_log_normalizer_from_checkpoint
from causica.datasets.tensordict_utils import expand_tensordict_groups
from causica.graph.evaluation_metrics import adjacency_f1, orientation_f1
from causica.lightning.modules.deci_module import DECIModule
from causica.training.evaluation import list_logsumexp, list_mean

logger = logging.getLogger(__name__)


class ExampleDECIModule(DECIModule):
    NUM_GRAPH_SAMPLES = 5

    def prepare_data(self) -> None:
        super().prepare_data()
        datamodule = getattr(self.trainer, "datamodule", None)
        if not isinstance(datamodule, ExampleDataModule):
            raise ValueError("The trainer must have a ExampleDataModule.")

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        """Create a callback for the auglag callback."""
        callback = super().configure_callbacks()
        callbacks_list = [callback] if isinstance(callback, pl.Callback) else list(callback)
        return callbacks_list + [LearningRateMonitor(logging_interval="step")]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Ensure correct loading of the normalizer from the checkpoint."""
        self.normalizer = infer_compatible_log_normalizer_from_checkpoint(checkpoint["state_dict"])

    def setup(self, stage: Optional[str] = None):
        if self.is_setup:
            return  # Already setup

        super().setup(stage)

        if stage is None:
            # Skip the rest as we are loading from a checkpoint
            return

        assert self.variable_types is not None

        if not hasattr(self.trainer, "datamodule"):
            raise ValueError("Trainer must have a datamodule.")

        datamodule = getattr(self.trainer, "datamodule", None)
        assert isinstance(datamodule, ExampleDataModule)

        if datamodule.normalizer is None:
            raise ValueError("Data module must have a normalizer.")
        self.normalizer = datamodule.normalizer
        self.variable_names: dict[str, list[str]] = datamodule.column_names
        # feature level metrics
        self.metrics_wrapper_all = torch.nn.ModuleDict(
            {
                "validation": MultitaskWrapper(
                    {
                        "Revenue": MetricCollection(
                            {"mape": MeanAbsolutePercentageError(), "mae": MeanAbsoluteError()}, postfix=".Revenue"
                        ),
                        "Discount": MetricCollection(
                            {"accuracy": Accuracy(task="binary"), "f1": F1Score(task="binary")}, postfix=".Discount"
                        ),
                    }
                ),
                "test": MultitaskWrapper(
                    {
                        "Revenue": MetricCollection(
                            {"mape": MeanAbsolutePercentageError(), "mae": MeanAbsoluteError()}, postfix=".Revenue"
                        ),
                        "Discount": MetricCollection(
                            {"accuracy": Accuracy(task="binary"), "f1": F1Score(task="binary")}, postfix=".Discount"
                        ),
                    }
                ),
            }
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        _ = dataloader_idx
        dataset_size = self.trainer.datamodule.dataset_valid.batch_size  # type: ignore
        return self.observational_evaluation_step(
            batch,
            batch_idx,
            log_prefix="valid_eval",
            dataset_size=dataset_size,
            metrics_wrapper_dict=self.metrics_wrapper_all["validation"],
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        _ = dataloader_idx
        dataset_size = self.trainer.datamodule.dataset_test.batch_size  # type: ignore
        return self.observational_evaluation_step(
            batch,
            batch_idx,
            log_prefix="test",
            dataset_size=dataset_size,
            metrics_wrapper_dict=self.metrics_wrapper_all["test"],
        )

    def on_test_epoch_start(self) -> None:
        if getattr(self, "trainer", None) is not None and hasattr(self.trainer, "datamodule"):
            true_adj_matrix = self.trainer.datamodule.true_adj
            self.test_graph(true_adj_matrix)

        return super().on_test_epoch_start()

    def test_graph(self, true_adj_matrix):
        """Evaluate the graph reconstruction performance of the model against the true graph."""
        sem_samples = self.sem_module().sample(torch.Size([self.NUM_GRAPH_SAMPLES]))
        graph_samples = [sem.graph for sem in sem_samples]

        true_adj_matrix = true_adj_matrix.to(graph_samples[0].device)

        adj_f1 = list_mean([adjacency_f1(true_adj_matrix, graph) for graph in graph_samples]).item()
        orient_f1 = list_mean([orientation_f1(true_adj_matrix, graph) for graph in graph_samples]).item()
        graph_metrics = {"test.graph.adjacency.f1": adj_f1, "test.graph.orientation.f1": orient_f1}
        self.log_dict(graph_metrics, add_dataloader_idx=False, on_epoch=True)
        return graph_metrics

    @torch.no_grad()
    def observational_evaluation_step(
        self,
        batch: TensorDict,
        batch_idx: int,
        log_prefix: str = "metrics",
        dataset_size: torch.Size = torch.Size([1]),
        metrics_wrapper_dict: Optional[MultitaskWrapper] = None,
    ):
        """Evaluate the log prob of the model for one batch using multiple graph samples.
        Args:
            metrics_wrapper_dict: dict containing MultiTaskWrapper obj for evaluation.
        """
        _ = batch_idx

        assert len(log_prefix) > 0, "log_prefix must be a non-empty string"

        # This is required for binary variables
        batch = batch.apply(lambda t: t.to(torch.float32))
        sems = self.sem_module().sample(torch.Size([self.NUM_GRAPH_SAMPLES]))
        assert len(dataset_size) == 1, "Only one batch size is supported"

        # estimate log prob for each sample using graph samples and report the mean over graphs
        test_logprob = list_logsumexp([sem.log_prob(batch) for sem in sems]) - np.log(self.NUM_GRAPH_SAMPLES)

        stacked: TensorDict = torch.stack([sem.func(batch, sem.graph) for sem in sems])
        mean_predictions = stacked.apply(lambda v: v.mean(axis=0), batch_size=batch.batch_size, inplace=False)

        assert self.normalizer is not None

        mean_predictions_orig = self.normalizer.inv(mean_predictions)
        observations_orig = self.normalizer.inv(batch)

        # group_variable_names
        mean_predictions_orig = expand_tensordict_groups(mean_predictions_orig, self.variable_names)
        observations_orig = expand_tensordict_groups(observations_orig, self.variable_names)
        mean_predictions = expand_tensordict_groups(mean_predictions, self.variable_names)

        assert metrics_wrapper_dict is not None
        metrics_wrapper_dict.forward(mean_predictions_orig.to_dict(), observations_orig.to_dict())
        self.log(
            f"{log_prefix}.test_LL",
            torch.sum(test_logprob, dim=-1).item() / dataset_size[0],
            reduce_fx=sum,
            add_dataloader_idx=False,
        )

        self.log_dict(
            {
                f"{log_prefix}.{key}": val
                for variable_metric in metrics_wrapper_dict.task_metrics.values()
                for key, val in variable_metric.items()
            },
            add_dataloader_idx=False,
            on_epoch=True,
        )
