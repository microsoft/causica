import os
from typing import List, Optional, cast

import pytorch_lightning as pl
import torch
import yaml
from fip.data_modules.numpy_tensor_data_module import NumpyTensorDataModule
from fip.methods.scm_learning import SCMLearning
from fip.task_utils.learnable_loss import LearnableGaussianLLH
from fip.tasks.amortization.leaf_prediction import LeafPrediction
from pytorch_lightning import Trainer
from torch import nn, optim

from causica.graph.evaluation_metrics import (
    adjacency_f1,
    orientation_f1,
    orientation_fallout_recall,
    orientation_precision_recall,
)


class SCMLearningPredLeaf(pl.LightningModule):
    """Lightning trainer for the task of causal discovery end-to-end with predicted leaf nodes."""

    def __init__(
        self,
        leaf_model_path: str,
        leaf_config_path: str,
        lr: float = 1e-4,
        weight_decay: float = 1e-10,
        d_model: int = 64,
        num_heads: int = 1,
        dim_key: int = 32,
        d_feedforward: int = 128,
        total_nodes: int = 10,
        total_layers: int = 1,
        dropout_prob: float = 0.0,
        mask_type: str = "triang",
        attn_type: str = "causal",
        cost_type: str = "dot_product",
        learnable_loss: bool = False,
        distributed: bool = False,
    ):
        """
        Args:
            lr: Learning rate for the optimizer
            weight_decay: Weight decay for the optimizer
            leaf_model_path: Path to the model for predicting the leaf nodes
            leaf_config_path: Path to the config file of the model for predicting the leaf nodes
            d_model: Embedding dimension used in the transformer model
            num_heads: Total number heads for self attention in the transformer model
            dim_key: Dimension of the key in the transformer model
            d_feedforward: Hidden dimension for feedforward layer in the transformer model
            total_nodes: Total number of nodes in the graph
            total_layers: Total self attention layers for the transformer model
            dropout_prob: Dropout probability for the transformer model
            mask_type: Type of the mask for the cross-attention layers
            attn_type: Type of attention for the cross-attention layers
            cost_type: Type of cost for the cross-attention layers
            learnable_loss: Flag to decide whether to use learnable loss
        """

        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.total_nodes = total_nodes

        if not os.path.isfile(leaf_model_path):
            raise ValueError(f"Leaf Model path {leaf_model_path} does not exist")

        if not os.path.isfile(leaf_config_path):
            raise ValueError(f"Leaf Config path {leaf_config_path} does not exist")

        self.leaf_pred_model = LeafPrediction.load_from_checkpoint(
            checkpoint_path=leaf_model_path
        )  # pylint: disable=all
        self.leaf_config_path = leaf_config_path

        # Thresholds for inferring the graph
        self.threshold: torch.Tensor
        self.register_buffer("threshold", torch.linspace(0.001, 1.0, steps=10))

        # Load Algorithm/Model
        self.method = SCMLearning(
            d_model=d_model,
            num_heads=num_heads,
            d_feedforward=d_feedforward,
            total_nodes=total_nodes,
            total_layers=total_layers,
            dropout_prob=dropout_prob,
            mask_type=mask_type,
            attn_type=attn_type,
            cost_type=cost_type,
            dim_key=dim_key,
        )
        self.loss_noise: nn.MSELoss | LearnableGaussianLLH
        if learnable_loss:
            self.loss_noise = LearnableGaussianLLH(max_seq_length=total_nodes)
        else:
            self.loss_noise = nn.MSELoss()

        self.distributed = distributed

        self.is_setup = False

        self.save_hyperparameters()

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
        self.standardize = datamodule.standardize
        self.mean_data = datamodule.train_data.mean_data.to(self.device)
        self.std_data = datamodule.train_data.std_data.to(self.device)

        data_dir_leaf = datamodule.data_dir
        with open(self.leaf_config_path, encoding="utf8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        batch_size_leaf = config["data"]["init_args"]["num_samples_used"]
        standardize_leaf = config["data"]["init_args"]["standardize"]

        datamodule_eval_leaf_pred = NumpyTensorDataModule(
            data_dir=data_dir_leaf,
            train_batch_size=batch_size_leaf,
            test_batch_size=batch_size_leaf,
            standardize=standardize_leaf,
            dod=True,
        )
        trainer = Trainer(devices=1, num_nodes=1)
        metrics = trainer.test(self.leaf_pred_model, datamodule=datamodule_eval_leaf_pred)
        print("Metrics", metrics)

        pred_sig = trainer.predict(self.leaf_pred_model, datamodule=datamodule_eval_leaf_pred)
        pred_sig = cast(List[torch.Tensor], pred_sig)
        self.pred_sig = pred_sig[0]
        print("Predicted topological ordering", self.pred_sig)

        self.is_setup = True

    def configure_optimizers(self):
        return optim.AdamW(
            self.method.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.95, 0.98),
            eps=1e-9,
        )

    def predict_graph(self, batch, threshold=0.1, mode="mean"):
        x, *_ = batch

        pred_sig = self.pred_sig
        x = x[:, pred_sig]

        # compute the mean graph
        with torch.enable_grad():
            mean_graph = self.method.get_aggregated_causal_graph(x, mode=mode)

        # compute the thresholded graph
        binary_graph = (mean_graph > threshold).float()

        return binary_graph

    def log_graph_metrics(self, batch, permuted_true_graph, name):
        batch_size = batch[0].shape[0]
        graph_pred = self.predict_graph(batch)
        test_adj_f1_pred = adjacency_f1(permuted_true_graph, graph_pred)
        test_orient_f1_pred = orientation_f1(permuted_true_graph, graph_pred)

        self.log(
            f"{name}_adj_f1_pred",
            test_adj_f1_pred,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.log(
            f"{name}_orient_f1_pred",
            test_orient_f1_pred,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        thresholds = torch.linspace(0.001, 1.0, steps=100).to(self.device)

        x = batch[0]
        pred_sig = self.pred_sig
        x = x[:, pred_sig]
        with torch.enable_grad():
            graphs = torch.unbind(self.method.get_threshold_causal_graph(x, threshold=thresholds), axis=-1)
        adj_f1_best = max(adjacency_f1(permuted_true_graph, graph) for graph in graphs)
        orient_f1_best = max(orientation_f1(permuted_true_graph, graph) for graph in graphs)

        self.log(
            f"{name}_adj_f1_best",
            adj_f1_best,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        self.log(
            f"{name}_orient_f1_best",
            orient_f1_best,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return graphs

    def log_pr_and_roc(self, permuted_true_graph, graphs, name):
        # compute auc for precision recall and recall fallout
        list_precision_pr = []
        list_recall_pr = []
        list_recall_roc = []
        list_fallout_roc = []
        for graph in graphs:
            test_orient_precision_pr, test_orient_recall_pr = orientation_precision_recall(permuted_true_graph, graph)
            test_orient_fallout_roc, test_orient_recall_roc = orientation_fallout_recall(permuted_true_graph, graph)
            list_precision_pr.append(test_orient_precision_pr)
            list_recall_pr.append(test_orient_recall_pr)
            list_recall_roc.append(test_orient_recall_roc)
            list_fallout_roc.append(test_orient_fallout_roc)

        # get tensor version
        list_recall_pr = torch.stack(list_recall_pr)
        list_precision_pr = torch.stack(list_precision_pr)

        # compute recall gain and precision gain
        num_nodes = permuted_true_graph.shape[-1]
        pos_ratio = torch.sum(permuted_true_graph) / (num_nodes * (num_nodes))
        list_precision_pr_gain = (list_precision_pr - pos_ratio) / ((1 - pos_ratio) * list_precision_pr)
        list_recall_pr_gain = (list_recall_pr - pos_ratio) / ((1 - pos_ratio) * list_recall_pr)

        index_reordering_pr_gain = list_recall_pr_gain.argsort()
        list_recall_pr_gain_ordered = list_recall_pr_gain[index_reordering_pr_gain]
        list_precision_pr_gain_ordered = list_precision_pr_gain[index_reordering_pr_gain]

        # consider only values in in [0,1] for both
        ind_recall = (list_recall_pr_gain_ordered >= 0) & (list_recall_pr_gain_ordered <= 1)
        ind_precision = (list_precision_pr_gain_ordered >= 0) & (list_precision_pr_gain_ordered <= 1)
        common_ind = ind_recall & ind_precision

        if torch.sum(common_ind) >= 0.5:
            list_recall_pr_ordered = list_recall_pr_gain_ordered[common_ind]
            list_precision_pr_ordered = list_precision_pr_gain_ordered[common_ind]

            # add (0, first precision) and (1,0)
            list_recall_pr_ordered = torch.cat(
                (torch.tensor([0.0]).to(self.device), list_recall_pr_ordered, torch.tensor([1.0]).to(self.device))
            )
            val_first_precision = list_precision_pr_ordered[0].item()
            list_precision_pr_ordered = torch.cat(
                (
                    torch.tensor([val_first_precision]).to(self.device),
                    list_precision_pr_ordered,
                    torch.tensor([0.0]).to(self.device),
                )
            )

            # compute auc for precision recall
            auc_precision_recall = torch.trapz(list_precision_pr_ordered, list_recall_pr_ordered)
            self.log(
                f"{name}_auc_precision_recall",
                auc_precision_recall,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
        else:
            print("No indices found for precision recall")

        # reorder fallout_roc and their associated recall_roc
        list_fallout_roc = torch.stack(list_fallout_roc)
        index_reordering_roc = list_fallout_roc.argsort()
        list_fallout_roc_ordered = list_fallout_roc[index_reordering_roc]
        list_recall_roc_ordered = torch.stack(list_recall_roc)[index_reordering_roc]

        if torch.sum(list_fallout_roc_ordered) > 0.0:

            # add (0,0) and (1,1) to the lists
            list_fallout_roc_ordered = torch.cat(
                (torch.tensor([0.0]).to(self.device), list_fallout_roc_ordered, torch.tensor([1.0]).to(self.device))
            )
            list_recall_roc_ordered = torch.cat(
                (torch.tensor([0.0]).to(self.device), list_recall_roc_ordered, torch.tensor([1.0]).to(self.device))
            )

            # compute auc for recall fallout
            auc_recall_fallout = torch.trapz(list_recall_roc_ordered, list_fallout_roc_ordered)
            self.log(
                f"{name}_auc_recall_fallout",
                auc_recall_fallout,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

    def test_step_obs(self, batch):
        x, n, true_graph, *_ = batch
        batch_size = x.shape[0]

        pred_sig = self.pred_sig
        x = x[:, pred_sig]
        n = n[:, pred_sig]

        # Infer the noise from observations
        n_hat = self.method.sample_to_noise(x)

        # Metric to compute test loss for noise as per the methods' objective
        test_loss = self.loss_noise(n_hat, torch.zeros_like(n_hat))
        self.log("test_loss", test_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)

        std_true_noise = torch.std(n, dim=0, keepdim=True)
        target_loss = self.loss_noise(std_true_noise, torch.zeros_like(std_true_noise).to(std_true_noise.device))
        self.log("target_test_loss", target_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)

        # Metric RMSE between predicted noise and the true noise in the standardized space
        test_rmse_noise_loss_per_sample = torch.sqrt(torch.mean((n - n_hat) ** 2, dim=1))
        test_rmse_noise_loss = torch.mean(test_rmse_noise_loss_per_sample)
        self.log(
            "test_rmse_noise_loss",
            test_rmse_noise_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        # Computing metrics to check the performance on causal discovery
        permuted_true_graph = true_graph[pred_sig, :][:, pred_sig]

        graphs = self.log_graph_metrics(batch, permuted_true_graph, "test")
        self.log_pr_and_roc(permuted_true_graph, graphs, "test")

    def test_step_cf(self, batch):
        f_data, cf_data, int_index, int_values, _ = batch
        batch_size = f_data.shape[0]

        pred_sig = self.pred_sig

        mean_data = self.mean_data[:, pred_sig]
        std_data = self.std_data[:, pred_sig]
        f_data = f_data[:, pred_sig]
        cf_data = cf_data[:, pred_sig]

        int_index = torch.argmax((pred_sig == int_index).float()).item()

        # Infer the noise from observations
        if self.standardize:
            cf_data_pred = self.method.ite_prediction(f_data, mean_data, std_data, int_index, int_values)
        else:
            mean_cf = torch.zeros_like(mean_data).to(mean_data.device)
            std_cf = torch.ones_like(std_data).to(std_data.device)
            cf_data_pred = self.method.ite_prediction(f_data, mean_cf, std_cf, int_index, int_values)

        # get their standardized version
        cf_data_std = (cf_data - mean_data) / std_data
        cf_data_pred_std = (cf_data_pred - mean_data) / std_data

        cf_root_mse_loss_per_sample = torch.sqrt(torch.mean((cf_data_pred - cf_data) ** 2, dim=1))
        cf_root_mse_loss = torch.mean(cf_root_mse_loss_per_sample)
        self.log(
            "test_rmse_cf_loss",
            cf_root_mse_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

        cf_root_mse_loss_std_per_sample = torch.sqrt(torch.mean((cf_data_pred_std - cf_data_std) ** 2, dim=1))
        cf_root_mse_loss_std = torch.mean(cf_root_mse_loss_std_per_sample)

        self.log(
            "test_rmse_cf_loss_std",
            cf_root_mse_loss_std,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

    def test_step(self, batch, _, dataloader_idx=0):
        match dataloader_idx:
            case 0:
                return self.test_step_obs(batch)
            case 1:
                return self.test_step_cf(batch)

    def validation_step(self, batch):
        x, n, true_graph, *_ = batch
        batch_size = x.shape[0]

        true_sig = self.pred_sig
        x = x[:, true_sig]
        n = n[:, true_sig]

        # Forward Pass
        n_hat = self.method.sample_to_noise(x)

        # Metric to compute test loss for noise as per the methods' objective
        val_loss = self.loss_noise(n_hat, torch.zeros_like(n_hat))
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)

        # Metric RMSE between predicted noise and the true noise in the standardized space
        val_rmse_noise_loss_per_sample = torch.sqrt(torch.mean((n - n_hat) ** 2, dim=1))
        val_rmse_noise_loss = torch.mean(val_rmse_noise_loss_per_sample)
        self.log(
            "val_rmse_noise_loss",
            val_rmse_noise_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        permuted_true_graph = true_graph[true_sig, :][:, true_sig]
        self.log_graph_metrics(batch, permuted_true_graph, "val")

    def training_step(self, batch):
        x, n, true_graph, *_ = batch
        batch_size = x.shape[0]

        pred_sig = self.pred_sig
        x = x[:, pred_sig]
        n = n[:, pred_sig]

        # Forward Pass
        n_hat = self.method.sample_to_noise(x)

        # train objective
        train_loss = self.loss_noise(n_hat, torch.zeros_like(n_hat))
        self.log(
            "train_loss",
            train_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

        std_true_noise = torch.std(n, dim=0, keepdim=True)
        target_train_loss = self.loss_noise(std_true_noise, torch.zeros_like(std_true_noise).to(std_true_noise.device))

        self.log(
            "target_train_loss",
            target_train_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

        # Computing metrics to check the performance on causal discovery
        permuted_true_graph = true_graph[pred_sig, :][:, pred_sig]
        self.log_graph_metrics(batch, permuted_true_graph, "train")

        mem_infos = torch.cuda.mem_get_info()
        used_memory = (mem_infos[1] - mem_infos[0]) / (1024**3)
        self.log(
            "Memory Info in GiB",
            used_memory,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

        return train_loss
