import pytorch_lightning as pl
import torch
from ema_pytorch import EMA
from torch import nn, optim

from cond_fip.models.amortized_models import AmortizedNoise


class EncoderTraining(pl.LightningModule):
    """Lightning trainer for the amortized learning of dataset embeddings."""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.95,
        weight_decay: float = 5e-4,
        use_scheduler: bool = False,
        linear_warmup_steps: int = 1000,
        scheduler_steps: int = 10_000,
        d_model: int = 64,
        num_heads: int = 1,
        num_layers: int = 4,
        d_ff: int = 128,
        dropout: float = 0.0,
        dim_key: int = 32,
        d_hidden_head: int = 1024,
        distributed: bool = False,
        with_ema: bool = True,
        ema_beta: float = 0.99,
        ema_update_every: int = 10,
    ) -> None:
        """
        Args:
            learning_rate: Learning rate for the optimizer
            beta1: First moment of Adam
            beta2: Second moment for Adam
            weight_decay: Weight decay strength for the optimizer
            use_scheduler: Whether to apply a scheduler for the learning rate
            linear_warmup_steps: Warmup for initializing the learning rate scheduler
            scheduler_steps: Total number of steps when using scheduler
            d_model: Embedding dimension used in the transformer model
            num_heads: Total number heads for self attention in the transformer model
            num_layers: Total self attention layers for the transformer model
            d_ff: Hidden dimension for feedforward layer in the transformer model
            dropout: Dropout probability for the transformer model
            dim_key: Dimension of queries, keys and values per head
            d_hidden_head: Hidden dimension of the head MLP
            distributed: Whether we use multiple gpus for training a single job
            with_ema: Whether to use exponential moving average of model's weight or not
            ema_beta: Hyperparameter for EMA
            ema_update_every: Number of iterations before updating EMA
        """

        super().__init__()

        self.method = AmortizedNoise(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            dim_key=dim_key,
            d_hidden_head=d_hidden_head,
        )
        self.with_ema = with_ema
        if self.with_ema:
            self.ema = EMA(self.method, beta=ema_beta, update_every=ema_update_every)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = (beta1, beta2)

        self.use_scheduler = use_scheduler
        self.linear_warmup_steps = linear_warmup_steps
        self.scheduler_steps = scheduler_steps

        self.distributed = distributed

        self.save_hyperparameters()

    def configure_optimizers(self):
        lr = self.learning_rate

        optimizer = optim.Adam(
            self.method.parameters(),
            lr=lr,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )

        if self.use_scheduler:

            def lr_lambda(current_step):
                if current_step < self.linear_warmup_steps:
                    return min(1.0, current_step / self.linear_warmup_steps)
                return (
                    0.5
                    * (
                        1
                        + torch.cos(
                            torch.Tensor([torch.pi])
                            * (current_step - self.linear_warmup_steps)
                            / (self.scheduler_steps - self.linear_warmup_steps)
                        )
                    ).item()
                )

            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                "interval": "step",
            }

            return [optimizer], [scheduler]

        return optimizer

    def update_mask(self, true_graph):
        self.method.encoder.dec_mask = true_graph.unsqueeze(1).unsqueeze(2)
        if self.with_ema:
            self.ema.ema_model.encoder.dec_mask = true_graph.unsqueeze(1).unsqueeze(2)

        self.method.encoder.sample_mask = None
        if self.with_ema:
            self.ema.ema_model.encoder.sample_mask = None

    def training_step(self, batch):
        (train_X, train_y, true_graph, *_) = batch
        batch_size = train_X.shape[0]

        self.update_mask(true_graph)

        # Main Training Loop
        y_pred = self.method(train_X)

        # rmse loss
        train_loss = torch.sqrt(torch.mean((y_pred - (train_X - train_y)) ** 2, dim=-1))
        train_loss = torch.mean(train_loss)
        self.log(
            "train_loss",
            train_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

        # mse loss
        mse_loss = nn.MSELoss()(y_pred, train_X - train_y)
        self.log(
            "mse_train_loss",
            mse_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

        # Update EMA
        if self.with_ema:
            self.ema.update()

        return train_loss

    def validation_step(self, batch, _):
        (val_X, val_y, true_graph, *_) = batch
        batch_size = val_X.shape[0]

        with torch.enable_grad():

            self.update_mask(true_graph)

            if self.with_ema:
                y_pred = self.ema.ema_model(val_X)
            else:
                y_pred = self.method(val_X)

            # rmse loss
            val_loss = torch.sqrt(torch.mean((y_pred - (val_X - val_y)) ** 2, dim=-1))
            val_loss = torch.mean(val_loss)
            self.log(
                "val_loss",
                val_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=self.distributed,
                batch_size=batch_size,
            )

            # mse loss
            mse_loss = nn.MSELoss()(y_pred, val_X - val_y)
            self.log(
                "mse_val_loss",
                mse_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=self.distributed,
                batch_size=batch_size,
            )
