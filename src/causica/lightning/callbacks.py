from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LightningArgumentParser, Namespace, SaveConfigCallback
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import STEP_OUTPUT

from causica.training.auglag import AugLagLossCalculator, AugLagLR


class AuglagLRCallback(pl.Callback):
    """Wrapper Class to make the Auglag Learning Rate Scheduler compatible with Pytorch Lightning"""

    def __init__(self, scheduler: AugLagLR, log_auglag: bool = False, disabled_epochs: Optional[set[int]] = None):
        """
        Args:
            scheduler: The auglag learning rate scheduler to wrap.
            log_auglag: Whether to log the auglag state as metrics at the end of each epoch.
        """
        self.scheduler = scheduler
        self._log_auglag = log_auglag
        self._disabled_epochs = disabled_epochs

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

        # Disable if we reached a disabled epoch - disable, otherwise make sure the scheduler is enabled
        if self._disabled_epochs and trainer.current_epoch in self._disabled_epochs:
            self.scheduler.disable(auglag_loss)
        else:
            self.scheduler.enable(auglag_loss)

        is_converged, convergence_reasons = self.scheduler.step(
            optimizer=optimizer,
            loss=auglag_loss,
            loss_value=outputs["loss"],
            lagrangian_penalty=outputs["constraint"],
        )

        # Notify trainer to stop if the auglag algorithm has converged
        if is_converged:
            pl_module.log(
                "auglag_inner_convergence_reason",
                convergence_reasons.inner_convergence_reason.value,
                on_epoch=True,
                rank_zero_only=True,
                prog_bar=False,
            )
            pl_module.log(
                "auglag_outer_convergence_reason",
                convergence_reasons.outer_convergence_reason.value,
                on_epoch=True,
                rank_zero_only=True,
                prog_bar=False,
            )
            trainer.should_stop = True

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        _ = trainer
        if self._log_auglag:
            auglag_state = {
                "num_lr_updates": self.scheduler.num_lr_updates,
                "outer_opt_counter": self.scheduler.outer_opt_counter,
                "step_counter": self.scheduler.step_counter,
                "outer_below_penalty_tol": self.scheduler.outer_below_penalty_tol,
                "outer_max_rho": self.scheduler.outer_max_rho,
                "last_best_step": self.scheduler.last_best_step,
                "last_lr_update_step": self.scheduler.last_lr_update_step,
            }
            pl_module.log_dict(auglag_state, on_epoch=True, rank_zero_only=True, prog_bar=False)


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
                temporary_config_path = str(Path(tmpdir) / self.config_filename)
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
