from typing import Any

import pytorch_lightning as pl
import torch
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

        self.scheduler.step(
            optimizer=optimizer,
            loss=auglag_loss,
            loss_value=outputs["loss"].item(),
            lagrangian_penalty=outputs["constraint"].item(),
        )
