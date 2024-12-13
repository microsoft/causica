import pytorch_lightning as pl
import torch
from ema_pytorch import EMA
from torch import nn, optim

from causica.functional_relationships.linear_functional_relationships import LinearFunctionalRelationships
from cond_fip.models.amortized_models import AmortizedEncoderDecoder
from cond_fip.tasks.encoder_training import EncoderTraining


class CondFiPTraining(pl.LightningModule):
    """Lightning trainer for the amortized learning of causal functional relationships (Cond-FiP)."""

    def __init__(
        self,
        encoder_model_path: str = "research_experiments/logging_enc/best_model-v6.ckpt",
        learning_rate: float = 1e-2,
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
        num_layers_dataset: int = 1,
        distributed: bool = False,
        with_true_target: bool = True,
        final_pair_only: bool = True,
        with_ema: bool = True,
        ema_beta: float = 0.99,
        ema_update_every: int = 10,
    ) -> None:
        """
        Args:
            encoder_model_path: Path to pretrained encoder for obtaining dataset representations
            learning_rate: Learning rate for the optimizer
            beta1: First moment of Adam
            beta2: Second moment of Adam
            weight_decay: Weight decay for the optimizer
            use_scheduler: Whether to use a scheduler for learning rate
            linear_warmup_steps: Number of steps for initializing the learning rate scheduler
            scheduler_steps: Total number of steps when using scheduler
            d_model: Embedding dimension used in the transformer model
            num_heads: Total number heads for self attention in the transformer model
            num_layers: Total self attention layers for the transformer model
            d_ff: Hidden dimension for feedforward layer in the transformer model
            dropout: Dropout probability for the transformer model
            dim_key: Dimension of queries, keys and values per head
            num_layers_dataset: Total number of self attention layers for dataset representation transfomer model
            distributed: Decide whether we use multiple gpus for training a single job
            with_true_target: Whether we use ground truth target or use pretrained encoder to obtain targets
            final_pair_only: Whether to use the final pair (input, target) from the fixed-point trajectory for training
            with_ema: Decide whether to use exponential moving average of model's weight or not
            ema_beta: Hyperparameter for EMA
            ema_update_every: Number of iterations before updating EMA
        """

        super().__init__()

        self.encoder_lightning = EncoderTraining.load_from_checkpoint(encoder_model_path)  # pylint: disable=E1120

        self.encoder_with_ema = self.encoder_lightning.with_ema
        if self.encoder_with_ema:
            self.encoder = self.encoder_lightning.ema.ema_model
        else:
            self.encoder = self.encoder_lightning.method

        self.encoder.eval()
        # remove gradient computation
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.method = AmortizedEncoderDecoder(
            d_model=d_model,
            num_heads=num_heads,
            dim_key=dim_key,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            d_data_emb=self.encoder.d_model,
            num_layers_dataset=num_layers_dataset,
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

        self.with_true_target = with_true_target
        self.final_pair_only = final_pair_only

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
        self.encoder.encoder.dec_mask = true_graph.unsqueeze(1).unsqueeze(2)
        self.method.dec_mask = true_graph.unsqueeze(1).unsqueeze(2)
        if self.with_ema:
            self.ema.ema_model.dec_mask = true_graph.unsqueeze(1).unsqueeze(2)

        self.encoder.encoder.sample_mask = None

    def training_step(self, batch):
        (train_X, train_y, true_graph, *_) = batch

        self.update_mask(true_graph)

        # Obtain dataset embedding and log the encoder error
        dataset_embedded, rmse_err_enc, mse_err_enc = self.get_dataset_embedding(train_X, train_y)
        self.log(
            "train_rmse_noise_loss_std",
            rmse_err_enc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=train_X.shape[0],
        )

        self.log(
            "train_mse_noise_loss_std",
            mse_err_enc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=train_X.shape[0],
        )

        # Compute (input, target) pairs from true SCM
        dec_input, dec_target = self.compute_path_and_target(
            batch,
            final_pair_only=self.final_pair_only,
        )

        # Predictions
        dec_pred = self.method(dec_input, dataset_embedded)

        # compute the loss
        train_loss = torch.sqrt(nn.MSELoss(reduction="none")(dec_pred, dec_target).mean(-1).mean(-1)).mean()
        self.log(
            "train_loss",
            train_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=dec_input.shape[0],
        )

        # Update EMA
        if self.with_ema:
            self.ema.update()

        return train_loss

    def validation_step(self, batch):
        (val_X, val_y, true_graph, *_) = batch

        self.update_mask(true_graph)

        # Obtain dataset embedding and log the encoder error
        dataset_embedded, rmse_err_enc, mse_err_enc = self.get_dataset_embedding(val_X, val_y)
        self.log(
            "val_rmse_noise_loss_std",
            rmse_err_enc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=val_X.shape[0],
        )
        self.log(
            "val_mse_noise_loss_std",
            mse_err_enc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=val_X.shape[0],
        )

        # Compute (input, target) pairs from true SCM
        dec_input, dec_target = self.compute_path_and_target(
            batch,
            final_pair_only=self.final_pair_only,
        )

        # Predict using the decoder on the input points
        if self.with_ema:
            dec_pred = self.ema.ema_model(dec_input, dataset_embedded)
        else:
            dec_pred = self.method(dec_input, dataset_embedded)

        val_loss = torch.sqrt(nn.MSELoss(reduction="none")(dec_pred, dec_target).mean(-1).mean(-1)).mean()
        # val loss
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=dec_input.shape[0],
        )

        # mse loss
        mse_loss = nn.MSELoss()(dec_pred, dec_target)
        self.log(
            "mse_val_loss",
            mse_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=dec_input.shape[0],
        )

        # rmse loss
        rmse_loss = torch.sqrt(torch.mean((dec_pred - dec_target) ** 2, dim=-1))
        rmse_loss = torch.mean(rmse_loss)
        self.log(
            "rmse_val_loss",
            rmse_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=dec_input.shape[0],
        )

    def test_step(self, batch, _):
        (val_X, val_y, true_graph, *_) = batch

        self.update_mask(true_graph)

        # Obtain dataset embedding and log the encoder error
        dataset_embedded, rmse_err_enc, mse_err_enc = self.get_dataset_embedding(val_X, val_y)
        self.log(
            "test_rmse_noise_loss_std",
            rmse_err_enc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=val_X.shape[0],
        )
        self.log(
            "test_mse_noise_loss_std",
            mse_err_enc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=val_X.shape[0],
        )

        # Compute (input, target) pairs from true SCM
        dec_input, dec_target = self.compute_path_and_target(
            batch,
            final_pair_only=self.final_pair_only,
        )

        # Predict using the decoder on the input points
        if self.with_ema:
            dec_pred = self.ema.ema_model(dec_input, dataset_embedded)
        else:
            dec_pred = self.method(dec_input, dataset_embedded)

        test_loss = torch.sqrt(nn.MSELoss(reduction="none")(dec_pred, dec_target).mean(-1).mean(-1)).mean()
        self.log(
            "test_loss",
            test_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=dec_input.shape[0],
        )

        # mse loss
        mse_loss = nn.MSELoss()(dec_pred, dec_target)
        self.log(
            "mse_test_loss",
            mse_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=dec_input.shape[0],
        )

        # rmse loss
        rmse_loss = torch.sqrt(torch.mean((dec_pred - dec_target) ** 2, dim=-1))
        rmse_loss = torch.mean(rmse_loss)
        self.log(
            "rmse_test_loss",
            rmse_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=dec_input.shape[0],
        )

        # test generation of new points
        self.test_generation_new_points(val_X, val_y, with_true_noise=True, dataset_embedded=dataset_embedded)
        self.test_generation_new_points(val_X, val_y, with_true_noise=False, dataset_embedded=dataset_embedded)

        # test counterfactuals
        self.test_prediction_counterfactual(batch, dataset_embedded=dataset_embedded)

    def get_dataset_embedding(self, data_X, data_y):
        """Provides dataset embeddings using the pretrained encoder on a split of the dataset
        Args:
            data_X: expected shape (batch_size, num_samples, max_seq_length)
            data_y: expected shape (batch_size, num_samples, max_seq_length)

        Returns:
            dataset_embdeed: expected shape (batch_size, num_emb_samples, max_seq_length)
            rmse_err_enc: RMSE noise prediction error
            mse_err_enc: MSE noise prediction error

        """
        num_emb_samples = int(data_X.shape[1] / 2)
        with torch.no_grad():
            dataset_embedded = self.encoder.compute_encoding(data_X[:, :num_emb_samples, :])
            pred_enc = self.encoder.compute_proj(dataset_embedded)
            target_enc = data_X[:, :num_emb_samples, :] - data_y[:, :num_emb_samples, :]

            # rmse noise prediction
            rmse_err_enc = torch.sqrt(torch.mean((target_enc - pred_enc) ** 2, dim=-1))
            rmse_err_enc = torch.mean(rmse_err_enc)

            # mse noise prediction
            mse_err_enc = nn.MSELoss()(pred_enc, target_enc)

        return dataset_embedded, rmse_err_enc, mse_err_enc

    def compute_path_and_target(self, batch, final_pair_only=True):
        """Samples the (inputs, targets) for training cond-FiP from the fixed-point iterations.
        Args:
            batch: Batch from the lightning dataset loader
            final_pair_only: Whether  to sample from the last fixed-point iteration, or randomly from intermediate fixed-points iterations.

        Returns:
            inputs to the cond-FiP; expected shape (batch_size, num_emb_samples, max_seq_length)
            targets corresponding to the inputs for cond-FiP; expected shape (batch_size, num_emb_samples, max_seq_length)

        """
        (data_X, data_y, *_) = batch
        total_datasets = data_X.shape[0]
        num_emb_samples = int(data_X.shape[1] / 2)
        total_nodes = data_X.shape[-1]

        with torch.no_grad():
            if final_pair_only:
                # take only the final fixed-point equation
                inputs = data_X.clone()
                targets = inputs - data_y.clone()
            else:
                idx_path = torch.randint(0, total_nodes, (total_datasets,))

                list_inputs = []
                list_targets = []
                X_up_true_path = data_y.clone()
                for k in range(total_nodes):
                    # Update as per the next iterate given by the true SCM
                    list_inputs.append(X_up_true_path)
                    X_target = self.compute_true_func(batch, X_up_true_path)
                    list_targets.append(X_target)
                    X_up_true_path = X_target + data_y

                inputs = torch.zeros_like(data_X)
                targets = torch.zeros_like(data_y)
                for k in range(total_datasets):
                    # List_inputs is of shape: Path length * Num Datasets * Sample Size * Total Nodes
                    # For each dataset, select corresponding to the random iterate give by idx_path
                    curr_path = idx_path[k]
                    inputs[k] = list_inputs[curr_path][k]
                    targets[k] = list_targets[curr_path][k]

            # Take the second half of the data to obtain input points
            if self.with_true_target:
                return (
                    inputs[:, num_emb_samples:, :],
                    targets[:, num_emb_samples:, :],
                )

            if final_pair_only:
                # with predicted target as per the learned encoder
                return (
                    inputs[:, num_emb_samples:, :],
                    self.encoder(inputs[:, num_emb_samples:, :]),
                )

            raise NotImplementedError("When with_true_target is False, only num_path == 1 is implemented")

    def test_generation_new_points(self, val_X, val_y, with_true_noise=False, dataset_embedded=None):
        """Starting from noise n, test whether Cond-FiP can generate observational data x.
        Args:
            val_X: expected shape (batch_size, num_samples, max_seq_length)
            val_y: expected shape (batch_size, num_samples, max_seq_length)
            with_true_noise: Whether we utilize the true noise variables or the inferred noise variables
            dataset_embedded: expected shape (batch_size, num_emb_samples, max_seq_length)
        """
        num_emb_samples = int(val_X.shape[1] / 2)

        with torch.no_grad():
            val_X_h2 = val_X[:, num_emb_samples:, :]

            if with_true_noise:
                val_y_h2 = val_y[:, num_emb_samples:, :]
            else:
                val_y_h2 = val_X_h2 - self.encoder(val_X_h2)

            curr_gen = val_y_h2.clone()
            for _ in range(val_y_h2.shape[-1]):
                # Apply the learned functional mechanisms via the decoder
                if self.with_ema:
                    curr_gen = self.ema.ema_model(curr_gen, dataset_embedded)
                else:
                    curr_gen = self.method(curr_gen, dataset_embedded)

                # Add the noise as part of the additive SCM to obtain next iterate for fixed-poin problem
                curr_gen += val_y_h2

            test_rmse_generation_loss_std = torch.sqrt(torch.mean((curr_gen - val_X_h2) ** 2, dim=-1))
            test_rmse_generation_loss_std = test_rmse_generation_loss_std.mean()
            rmse_name_log = (
                "test_rmse_generation_loss_std" if with_true_noise else "test_rmse_generation_loss_std_fake_noise"
            )
            self.log(
                rmse_name_log,
                test_rmse_generation_loss_std,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=self.distributed,
                batch_size=val_y_h2.shape[0],
            )

            test_mse_generation_loss_std = nn.MSELoss()(curr_gen, val_X_h2)
            mse_name_log = (
                "test_mse_generation_loss_std" if with_true_noise else "test_mse_generation_loss_std_fake_noise"
            )
            self.log(
                mse_name_log,
                test_mse_generation_loss_std,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
                sync_dist=self.distributed,
                batch_size=val_y_h2.shape[0],
            )

    def test_prediction_counterfactual(self, batch, dataset_embedded=None):
        (val_X, *_) = batch

        num_interventions = val_X.shape[-1]
        loss_cf_w = 0.0
        loss_cf_wo = 0.0
        for _ in range(num_interventions):
            tensor_int_idx, tensor_int_val = self.generate_one_intervention(batch)

            # with true noise
            loss_cf_w_curr = self.compute_error_counterfactual(
                batch, tensor_int_idx, tensor_int_val, with_true_noise=True, dataset_embedded=dataset_embedded
            )
            loss_cf_w += loss_cf_w_curr

            # with fake noise
            loss_cf_wo_curr = self.compute_error_counterfactual(
                batch, tensor_int_idx, tensor_int_val, with_true_noise=False, dataset_embedded=dataset_embedded
            )
            loss_cf_wo += loss_cf_wo_curr

        loss_cf_w = loss_cf_w / num_interventions
        loss_cf_wo = loss_cf_wo / num_interventions

    def compute_error_counterfactual(self, batch, int_idx, int_val, with_true_noise=False, dataset_embedded=None):
        """Starting from observed data, test whether Cond-Fip can generate counterfactual data.
        Args:
            batch: Batch from the lightning dataset loader
            int_idx: Index of the causal variable to be intervened
            int_val: Value of the causal variable post hard intervention
            with_true_noise: Whether we utilize the true noise variables or the inferred noise variables
            dataset_embedded: expected shape (batch_size, num_emb_samples, max_seq_length)

        Returns:
            MSE reconstruction loss in countefactual generation

        """
        (val_X, val_y, *_) = batch
        num_emb_samples = int(val_X.shape[1] / 2)

        # compute true counterfactuals
        cf_true = self.generate_counterfactual(batch, int_idx, int_val)
        cf_true_h2 = cf_true[:, num_emb_samples:, :]

        with torch.no_grad():
            # embed all the data
            val_X_h2 = val_X[:, num_emb_samples:, :]

            if with_true_noise:
                val_y_h2 = val_y[:, num_emb_samples:, :]
                noise = val_y_h2
            else:
                noise = val_X_h2 - self.encoder(val_X_h2)

            cf_pred_h2 = noise.clone()
            for _ in range(val_y.shape[-1]):
                if self.with_ema:
                    cf_pred_h2 = self.ema.ema_model(cf_pred_h2, dataset_embedded)
                else:
                    cf_pred_h2 = self.method(cf_pred_h2, dataset_embedded)

                cf_pred_h2 += noise
                cf_pred_h2 = self.apply_intervention(int_idx, int_val, cf_pred_h2)

        loss_rmse_cf_per_sample = torch.sqrt(torch.mean((cf_pred_h2 - cf_true_h2) ** 2, dim=-1))
        loss_rmse_cf = loss_rmse_cf_per_sample.mean()

        rmse_name_log = "test_rmse_cf_loss" if with_true_noise else "test_rmse_cf_loss_fake_noise"
        self.log(
            rmse_name_log,
            loss_rmse_cf,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=val_X.shape[0],
        )

        loss_mse_cf = nn.MSELoss()(cf_pred_h2, cf_true_h2)
        mse_name_log = "test_mse_cf_loss" if with_true_noise else "test_mse_cf_loss_fake_noise"
        self.log(
            mse_name_log,
            loss_mse_cf,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=val_X.shape[0],
        )

        return loss_mse_cf

    def compute_true_func(self, batch, points):
        (_, _, true_graph, *metadata) = batch

        mean_data = metadata[0]
        std_data = metadata[1]
        list_func = metadata[-1]

        f_val = self.from_list_func_to_tensor(list_func, points, true_graph.transpose(-2, -1), mean_data, std_data)
        return f_val

    def from_list_func_to_tensor(self, list_func, val_X, true_graph, mean_data, std_data):
        targets = []
        val_X_unnormalized = val_X * std_data + mean_data
        for num_func, curr_func in enumerate(list_func):
            if isinstance(curr_func, LinearFunctionalRelationships):
                target_val = curr_func.linear_map(val_X_unnormalized[num_func], true_graph[num_func])
            else:
                target_val = curr_func.non_linear_map(val_X_unnormalized[num_func], true_graph[num_func])
            target_val = (target_val.unsqueeze(0) - mean_data[num_func]) / std_data[num_func]
            targets.append(target_val)
        return torch.cat(targets, dim=0)

    def compute_true_fixed_point(self, batch):
        (val_X, val_y, *_) = batch

        with torch.no_grad():
            f_val = val_y
            list_true_preds = [val_y]
            for _ in range(val_X.shape[-1]):
                f_val = self.compute_true_func(batch, f_val)
                f_val = f_val + val_y
                list_true_preds.append(f_val)

        return f_val, list_true_preds

    def generate_counterfactual(self, batch, int_idx, int_val):
        (_, val_y, *_) = batch

        with torch.no_grad():
            f_val = val_y.clone()
            for _ in range(val_y.shape[-1]):
                f_val = self.compute_true_func(batch, f_val)
                f_val = f_val + val_y
                f_val = self.apply_intervention(int_idx, int_val, f_val)

        return f_val

    def apply_intervention(self, tensor_int_idx, tensor_int_val, points):
        list_cf = []
        size = len(tensor_int_idx)

        for k in range(size):
            idx_curr = tensor_int_idx[k].item()
            val_curr = tensor_int_val[k].item()
            points[k][..., idx_curr] = val_curr
            list_cf.append(points[k])

        return torch.stack(list_cf, dim=0)

    def generate_one_intervention(self, batch, type_treatment="quantile"):
        (val_X, *_) = batch
        size = len(val_X)

        # generate random index and values for intervention
        list_int_idx = []
        list_int_val = []
        for k in range(size):
            int_idx = torch.randint(0, val_X[k].shape[-1], (1,)).item()
            list_int_idx.append(int_idx)

            if type_treatment == "quantile":
                min_val = torch.quantile(val_X[k][:, int_idx], 0.1).item()
                max_val = torch.quantile(val_X[k][:, int_idx], 0.9).item()
            else:
                min_val = val_X[k][:, int_idx].min()
                max_val = val_X[k][:, int_idx].max()

            int_val = torch.rand((1,)).to(val_X.device) * (max_val - min_val) + min_val
            list_int_val.append(int_val.item())

        # from list to tensor
        tensor_int_idx = torch.tensor(list_int_idx, device=val_X.device).unsqueeze(-1)
        tensor_int_val = torch.tensor(list_int_val, device=val_X.device).unsqueeze(-1)

        return tensor_int_idx, tensor_int_val
