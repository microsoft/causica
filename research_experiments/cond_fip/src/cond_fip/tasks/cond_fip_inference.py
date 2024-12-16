from typing import Optional

import pytorch_lightning as pl
import torch
from fip.data_modules.numpy_tensor_data_module import NumpyTensorDataModule
from torch import nn

from cond_fip.tasks.cond_fip_training import CondFiPTraining


class CondFiPInference(pl.LightningModule):
    """Lightning module for evaluation of Cond-FiP."""

    def __init__(
        self,
        enc_dec_model_path: str = "research_experiments/logging_enc/best_model-v6.ckpt",
    ) -> None:
        """
        Args:
            enc_dec_model_path: Path to pretrained Cond-FiP model
        """

        super().__init__()

        self.enc_dec = CondFiPTraining.load_from_checkpoint(enc_dec_model_path)  # pylint: disable=E1120

        self.encoder = self.enc_dec.encoder

        self.with_ema = self.enc_dec.with_ema
        if self.with_ema:
            self.decoder = self.enc_dec.ema.ema_model
        else:
            self.decoder = self.enc_dec.method

        self.encoder.eval()
        self.decoder.eval()

        # remove gradient computation in encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # remove gradient computation in decoder
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.is_setup = False

    def setup(self, stage: Optional[str] = None):
        _ = stage
        if self.is_setup:
            return  # Already setup

        datamodule = getattr(self.trainer, "datamodule", None)
        if not isinstance(datamodule, NumpyTensorDataModule):
            raise TypeError(
                f"Incompatible data module {datamodule}, requires a NumpyTensorDataModule but is "
                f"{type(datamodule).mro()}"
            )
        # get the training batches
        train_batches = datamodule.train_dataloader()

        # take only the first batch to compute the embedding
        train_batch = next(iter(train_batches))
        # apply the transformation on the training batch
        train_batch = datamodule.on_before_batch_transfer(train_batch, 0)
        train_X, train_y, true_graph, *_ = train_batch
        train_X, train_y, true_graph = train_X.to(self.device), train_y.to(self.device), true_graph.to(self.device)

        # save train_X, train_y
        self.train_X = train_X
        self.train_y = train_y

        self.update_mask(true_graph)

        self.is_setup = True

    def update_mask(self, true_graph):
        self.encoder.encoder.dec_mask = true_graph.unsqueeze(1).unsqueeze(2)
        self.decoder.dec_mask = true_graph.unsqueeze(1).unsqueeze(2)

        self.encoder.encoder.sample_mask = None

    def test_step(self, batch, _, dataloader_idx=0):
        match dataloader_idx:
            case 0:
                # validation step
                self.test_obs(batch, with_true_target=True)
                self.test_obs(batch, with_true_target=False)

                # test generation of new points
                self.test_generation_new_points(batch, with_true_noise=True)
                self.test_generation_new_points(batch, with_true_noise=False)

            case 1:
                # test counterfactuals
                self.test_prediction_counterfactual(batch, with_true_noise=True)
                self.test_prediction_counterfactual(batch, with_true_noise=False)

    def test_obs(self, batch, with_true_target=False):
        """Starting from observational data, test whether Cond-FiP can infer noise variables and also reconstruct observational data.
        Args:
            batch: Batch from the lightning dataset loader
            with_true_noise: Whether we utilize the true noise variables or the inferred noise variables
        """
        (val_X, val_y, *_) = batch

        # get current meand and std
        curr_data = torch.cat([self.train_X, val_X], dim=1)
        mean_data = curr_data.mean(dim=1, keepdim=True)
        std_data = curr_data.std(dim=1, keepdim=True)

        # standardize the data
        train_X = (self.train_X - mean_data) / std_data
        train_y = self.train_y / std_data
        val_X = (val_X - mean_data) / std_data
        val_y = val_y / std_data

        with torch.enable_grad():
            dataset_embedded = self.encoder.compute_encoding(train_X)

            # rmse noise prediction
            pred_enc = self.encoder.compute_proj(dataset_embedded)
            target_enc = train_X - train_y
            err_enc = torch.sqrt(torch.mean((target_enc - pred_enc) ** 2, dim=-1))
            err_enc = torch.mean(err_enc)
            self.log(
                "test_rmse_noise_loss_std",
                err_enc,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=train_X.shape[0],
            )

            err_mse_enc = nn.MSELoss()(target_enc, pred_enc)
            self.log(
                "test_mse_noise_loss_std",
                err_mse_enc,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=train_X.shape[0],
            )

            if with_true_target:
                target = val_X - val_y
            else:
                target = self.encoder(val_X)

            x_pred = self.decoder(val_X, dataset_embedded)

            test_loss = torch.sqrt(nn.MSELoss(reduction="none")(x_pred, target).mean(-1).mean(-1)).mean()
            name_log = "test_loss" if with_true_target else "test_loss_predicted_target"
            self.log(
                name_log,
                test_loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                batch_size=x_pred.shape[0],
            )

            mse_loss = nn.MSELoss()(x_pred, target)
            name_log = "mse_test_loss" if with_true_target else "mse_test_loss_predicted_target"
            self.log(name_log, mse_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x_pred.shape[0])

    def test_generation_new_points(self, batch, with_true_noise=False):
        """Starting from noise n, test whether Cond-FiP can generate observational data x.
        Args:
            batch: Batch from the lightning dataset loader
            with_true_noise: Whether we utilize the true noise variables or the inferred noise variables
        """
        (val_X, val_y, *_) = batch
        batch_size = val_X.shape[0]

        # get current meand and std
        curr_data = torch.cat([self.train_X, val_X], dim=1)
        mean_data = curr_data.mean(dim=1, keepdim=True)
        std_data = curr_data.std(dim=1, keepdim=True)

        # standardize the data
        train_X = (self.train_X - mean_data) / std_data
        val_X = (val_X - mean_data) / std_data
        val_y = val_y / std_data

        with torch.no_grad():
            dataset_embedded = self.encoder.compute_encoding(train_X)

            if not with_true_noise:
                noise = val_X - self.encoder(
                    val_X
                )  # we can also concatenate val_X with train_X from set up to improve the prediction of the noise
            else:
                noise = val_y

            curr_gen = noise.clone()
            for _ in range(val_y.shape[-1]):
                curr_gen = self.decoder(curr_gen, dataset_embedded)
                curr_gen += noise

            loss_gen_rmse_per_sample = torch.sqrt(torch.mean((curr_gen - val_X) ** 2, dim=-1))
            loss_gen = loss_gen_rmse_per_sample.mean()
            name_log = (
                "test_rmse_generation_loss_std" if with_true_noise else "test_rmse_generation_loss_std_predicted_noise"
            )
            self.log(name_log, loss_gen, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)

            loss_gen_mse = nn.MSELoss()(curr_gen, val_X)
            name_log = (
                "test_mse_generation_loss_std" if with_true_noise else "test_mse_generation_loss_std_predicted_noise"
            )
            self.log(name_log, loss_gen_mse, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)

    def test_prediction_counterfactual(self, batch, with_true_noise=False):
        """Starting from observed data, test whether Cond-Fip can generate counterfactual data.
        Args:
            batch: Batch from the lightning dataset loader
            with_true_noise: Whether we utilize the true noise variables or the inferred noise variables
        """
        f_data, n_data, cf_data, int_index, int_values, true_graph = batch
        batch_size = f_data.shape[0]

        f_data = f_data.unsqueeze(0)
        n_data = n_data.unsqueeze(0)
        cf_data = cf_data.unsqueeze(0)
        true_graph = true_graph.unsqueeze(0)
        int_index = int_index.item()
        int_values = int_values[0].item()

        curr_data = torch.cat([self.train_X[0].unsqueeze(0), f_data], dim=1)
        mean_data = curr_data.mean(dim=1, keepdim=True)
        std_data = curr_data.std(dim=1, keepdim=True)

        f_data_norm = (f_data - mean_data) / std_data
        n_data_norm = n_data / std_data
        cf_data_norm = (cf_data - mean_data) / std_data
        int_values_norm = (int_values - mean_data[..., int_index]) / std_data[..., int_index]
        int_values_norm = int_values_norm.item()

        with torch.no_grad():
            dataset_embedded = self.encoder.compute_encoding(f_data_norm)

            if with_true_noise:
                noise_norm = n_data_norm
            else:
                noise_norm = f_data_norm - self.encoder(
                    f_data_norm
                )  # we can also concatenate f_data_norm with train_X from set up to improve the prediction of the noise

            cf_pred = noise_norm.clone()
            for _ in range(noise_norm.shape[-1]):
                cf_pred = self.decoder(cf_pred, dataset_embedded)
                cf_pred += noise_norm
                cf_pred[..., int_index] = int_values_norm

        loss_cf_rmse_per_sample = torch.sqrt(torch.mean((cf_pred - cf_data_norm) ** 2, dim=-1))
        loss_cf = loss_cf_rmse_per_sample.mean()

        name_log = "test_rmse_cf_loss" if with_true_noise else "test_rmse_cf_loss_predicted_noise"
        self.log(name_log, loss_cf, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)

        loss_cf_mse = nn.MSELoss()(cf_pred, cf_data_norm)
        name_log = "test_mse_cf_loss" if with_true_noise else "test_mse_cf_loss_predicted_noise"
        self.log(name_log, loss_cf_mse, prog_bar=True, on_step=True, on_epoch=True, batch_size=batch_size)
