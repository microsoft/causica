from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LightningArgumentParser, Namespace, SaveConfigCallback
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import STEP_OUTPUT

from causica.training.auglag import AugLagLossCalculator, AugLagLR


class AuglagLRCallback(pl.Callback):
    """Wrapper Class to make the Auglag Learning Rate Scheduler compatible with Pytorch Lightning"""

    def __init__(self, scheduler: AugLagLR):
        """
        Args:
            scheduler: The auglag learning rate scheduler to wrap
        """
        self.scheduler = scheduler

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        _ = trainer
        _ = batch
        _ = batch_idx
        assert isinstance(outputs, dict)
        optimizer = pl_module.optimizers()
        assert isinstance(optimizer, torch.optim.Optimizer)
        auglag_loss: AugLagLossCalculator = pl_module.auglag_loss  # type: ignore

        is_converged = self.scheduler.step(
            optimizer=optimizer,
            loss=auglag_loss,
            loss_value=outputs["loss"].item(),
            lagrangian_penalty=outputs["constraint"].item(),
        )

        # Notify trainer to stop if the auglag algorithm has converged
        if is_converged:
            trainer.should_stop = True


class MLFlowSaveConfigCallback(SaveConfigCallback):
    """Logs the config using MLFlow if there is an active run, otherwise saves locally as the superclass."""

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str = "config.yaml",
        multifile: bool = False,
    ) -> None:
        super().__init__(
            parser=parser, config=config, config_filename=config_filename, overwrite=True, multifile=multifile
        )

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: TrainerFn) -> None:  # type: ignore
        # Save the file on rank 0
        if trainer.is_global_zero and stage == TrainerFn.FITTING:
            with TemporaryDirectory() as tmpdir:
                temporary_config_path = str(Path(tmpdir) / (f"{stage.value}_" + self.config_filename))
                self.parser.save(
                    self.config,
                    temporary_config_path,
                    skip_none=False,
                    overwrite=self.overwrite,
                    multifile=self.multifile,
                )
                # AzureML throws a raw Exception if the artifact already exists, so we check the error message
                try:
                    mlflow.log_artifact(temporary_config_path)
                # pylint: disable=broad-exception-caught
                except Exception as e:
                    if "Resource Conflict" not in str(e.args):
                        raise e
