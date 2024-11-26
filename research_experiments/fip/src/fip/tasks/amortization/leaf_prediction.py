import math
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim

from fip.models.amortized_models import AmortizedLeaf
from fip.task_utils.leaf_functions import (
    decreasing_sig_to_perm,
    find_leaf_nodes,
    graph_to_perm_torch,
    remove_leaf_nodes,
    vote_leaf_predicition,
)


class LeafPrediction(pl.LightningModule):
    """Lightning trainer for the amortized noise learning

    In this task, we are predicting sequentially the leaves of the causal graph.
    At each step of the inner loop, a leaf is deleted.

    Attributes:
        learning_rate: Learning rate for the optimizer
        weight_decay: Weight decay for the optimizer
        method: Method for learning structural equation with fixed-point solvers
    """

    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        d_model: int,
        num_heads: int,
        dim_key: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        max_num_leaf: int,
        distributed: bool = False,
        elimination_type: str = "self",
        num_to_keep_training: Optional[int] | Optional[list[int]] = None,
    ) -> None:
        """
        Args:
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for the optimizer
            d_model: Embedding dimension used in the transformer model
            num_heads: Total number heads for self attention in the transformer model
            dim_key: Dimension of the key in the transformer model
            num_layers: Total attention layers for the transformer model
            d_ff: Hidden dimension for feedforward layer in the transformer model
            dropout: Dropout probability for the transformer model
            max_num_leaf: Maximum number of leaf nodes in the graph
            distributed: Whether to use distributed training
            elimination_type: Type of elimination to use for the training
            num_to_keep_training: number of leaf predicition to keep during training chosen at random, or a list of fixed element to keep during training
        """

        super().__init__()

        self.method = AmortizedLeaf(
            d_model=d_model,
            num_heads=num_heads,
            dim_key=dim_key,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_num_leaf=max_num_leaf,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.num_to_keep_training = num_to_keep_training
        self.distributed = distributed
        self.elimination_type = elimination_type

        self.save_hyperparameters()

    def configure_optimizers(self):

        datamodule = getattr(self.trainer, "datamodule", None)

        learning_rate = self.learning_rate
        if datamodule is not None:
            learning_rate *= math.sqrt(datamodule.train_batch_size)

        optimizer = optim.AdamW(
            self.method.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.95, 0.98),
            eps=1e-9,
        )

        return optimizer

    def generate_list_indices(self, tot_num_nodes: int):
        if isinstance(self.num_to_keep_training, list):
            ind_to_keep = [x for x in self.num_to_keep_training if -1 < x < tot_num_nodes - 1]
        elif isinstance(self.num_to_keep_training, int):
            to_keep = min(self.num_to_keep_training, tot_num_nodes - 1)
            ind_to_keep = torch.randperm(tot_num_nodes - 1)[:to_keep].tolist()
        else:
            ind_to_keep = list(range(tot_num_nodes - 1))

        return ind_to_keep

    def select_leaf_nodes(self, leaf_pred: torch.Tensor, leaf_nodes_prob: torch.Tensor, graph: torch.Tensor):
        argmax_ind = torch.argmax(leaf_pred, dim=-1)
        argmax_ind = argmax_ind.detach()
        argmax_ind_list = argmax_ind.tolist()

        if self.elimination_type == "random":
            ind_to_remove = torch.multinomial(leaf_nodes_prob, num_samples=1).squeeze(-1)
        elif self.elimination_type == "self":
            ind_to_remove = torch.multinomial(leaf_nodes_prob, num_samples=1).squeeze(-1)
            ind_self = torch.where(leaf_nodes_prob[torch.arange(len(argmax_ind_list)), argmax_ind_list].float() != 0.0)[
                0
            ]
            ind_to_remove[ind_self] = argmax_ind[ind_self]
        else:
            sig = graph_to_perm_torch(graph)
            ind_to_remove = sig[:, -1]

        return ind_to_remove

    def log_curr_true_loss(
        self,
        leaf_pred: torch.Tensor,
        leaf_nodes_prob: torch.Tensor,
        ind_node: int,
        batch_size: int,
        tot_num_nodes: int,
        name: str,
    ):
        argmax_ind = torch.argmax(leaf_pred, dim=-1)
        argmax_ind = argmax_ind.detach()

        argmax_ind_list = argmax_ind.tolist()
        curr_true_loss = torch.sum(
            leaf_nodes_prob[torch.arange(len(argmax_ind_list)), argmax_ind_list].float() == 0.0
        ) / len(argmax_ind_list)
        self.log(
            f"{name}_true_loss with num_nodes " + str(tot_num_nodes - ind_node),
            curr_true_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )
        return curr_true_loss

    def training_step(self, batch, _):
        train_X, _, graph, *_ = batch
        batch_size = train_X.shape[0]

        # choose the list of leaf nodes to keep during training
        tot_num_nodes = graph.shape[-1]
        ind_to_keep = self.generate_list_indices(tot_num_nodes)

        true_loss = 0.0
        train_loss = 0.0
        for k in range(tot_num_nodes - 1):
            leaf_pred = self.method(train_X)

            leaf_nodes_prob = find_leaf_nodes(graph).to(graph.device)
            log_prob = F.logsigmoid(leaf_pred)
            train_loss_element_wise = -(leaf_nodes_prob * log_prob + (1.0 - leaf_nodes_prob) * F.logsigmoid(-leaf_pred))
            train_loss_element_wise = train_loss_element_wise.mean(dim=-1)

            if k in ind_to_keep:
                train_loss += train_loss_element_wise.mean()

            curr_true_loss = self.log_curr_true_loss(leaf_pred, leaf_nodes_prob, k, batch_size, tot_num_nodes, "train")
            true_loss += curr_true_loss

            # select the indices and remove leaf nodes
            ind_to_remove = self.select_leaf_nodes(leaf_pred, leaf_nodes_prob, graph)
            graph, train_X = remove_leaf_nodes(graph, train_X, ind_to_remove)

        train_loss = train_loss / len(ind_to_keep)
        self.log(
            "train_loss",
            train_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

        true_loss = true_loss / (tot_num_nodes - 1)
        self.log(
            "train_true_loss",
            true_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

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

    def validation_step(self, batch, _):
        val_X, _, graph, *_ = batch
        batch_size = val_X.shape[0]
        tot_num_nodes = graph.shape[-1]

        true_loss = 0.0
        val_loss = 0.0
        for k in range(tot_num_nodes - 1):
            leaf_pred = self.method(val_X)

            leaf_nodes_prob = find_leaf_nodes(graph).to(graph.device)
            log_prob = F.logsigmoid(leaf_pred)
            val_loss_element_wise = -(leaf_nodes_prob * log_prob + (1.0 - leaf_nodes_prob) * F.logsigmoid(-leaf_pred))
            val_loss_element_wise = val_loss_element_wise.mean(dim=-1)
            val_loss += val_loss_element_wise.mean()

            curr_true_loss = self.log_curr_true_loss(leaf_pred, leaf_nodes_prob, k, batch_size, tot_num_nodes, "val")
            true_loss += curr_true_loss

            # remove the leaf node
            ind_to_remove = self.select_leaf_nodes(leaf_pred, leaf_nodes_prob, graph)
            graph, val_X = remove_leaf_nodes(graph, val_X, ind_to_remove)

        val_loss = val_loss / (tot_num_nodes - 1)
        self.log(
            "val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

        true_loss = true_loss / (tot_num_nodes - 1)
        self.log(
            "val_true_loss",
            true_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

    def test_step(self, batch, _):
        self.test_step_eval(batch, _)
        self.test_step_hard(batch, _)
        self.test_step_soft(batch, _)

    def test_step_eval(self, batch, _):
        test_X, _, graph, *_ = batch
        batch_size = test_X.shape[0]
        tot_num_nodes = graph.shape[-1]

        tot_true_loss = 0.0
        curr_true_loss_list = []
        for k in range(tot_num_nodes - 1):
            leaf_nodes_prob = find_leaf_nodes(graph).to(graph.device)
            leaf_pred = self.method(test_X)

            curr_true_loss = self.log_curr_true_loss(leaf_pred, leaf_nodes_prob, k, batch_size, tot_num_nodes, "test")
            tot_true_loss += curr_true_loss
            curr_true_loss_list.append(curr_true_loss)

            # remove the leaf node
            ind_to_remove = self.select_leaf_nodes(leaf_pred, leaf_nodes_prob, graph)
            graph, test_X = remove_leaf_nodes(graph, test_X, ind_to_remove)

        tot_true_loss = tot_true_loss / (tot_num_nodes - 1)
        self.log(
            "test_true_loss",
            tot_true_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

        self.log(
            "test_max_true_loss",
            max(curr_true_loss_list),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

    def test_step_hard(self, batch, _):
        test_X, _, graph, *_ = batch

        graph = graph[0].unsqueeze(0)
        batch_size = test_X.shape[0]
        tot_num_nodes = graph.shape[-1]

        ## generate the current topological ordering predicted by the leaf model
        leaf_decreasing_order = []
        true_loss = 0.0
        for _ in range(tot_num_nodes - 1):
            perm_pred = self.method(test_X)

            ## hard-sum
            argmax_ind = torch.argmax(perm_pred, dim=-1).tolist()
            ind_leaf = vote_leaf_predicition(argmax_ind)

            leaf_decreasing_order.append(ind_leaf)

            leaf_nodes_prob = find_leaf_nodes(graph).to(graph.device)
            curr_true_loss = (leaf_nodes_prob[0, ind_leaf] == 0.0).float()
            true_loss += curr_true_loss

            # remove the leaf node
            ind_to_remove = torch.Tensor([ind_leaf for k in range(batch_size)]).long()
            graph, test_X = remove_leaf_nodes(graph, test_X, ind_to_remove)

        true_loss = true_loss / (tot_num_nodes - 1)
        self.log(
            "test_true_loss_hard",
            true_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

    def test_step_soft(self, batch, _):
        test_X, _, graph, *_ = batch

        graph = graph[0].unsqueeze(0)
        batch_size = test_X.shape[0]
        tot_num_nodes = graph.shape[-1]

        ## generate the current topological ordering predicted by the leaf model
        leaf_decreasing_order = []
        true_loss = 0.0
        for _ in range(tot_num_nodes - 1):
            perm_pred = self.method(test_X)

            ## soft-sum
            cumulate_probs = torch.sigmoid(perm_pred).mean(dim=0)
            ind_leaf = torch.argmax(cumulate_probs, dim=-1).item()

            leaf_decreasing_order.append(ind_leaf)

            leaf_nodes_prob = find_leaf_nodes(graph).to(graph.device)
            curr_true_loss = (leaf_nodes_prob[0, ind_leaf] == 0.0).float()
            true_loss += curr_true_loss

            # remove the leaf node
            ind_to_remove = torch.Tensor([ind_leaf for k in range(batch_size)]).long()
            graph, test_X = remove_leaf_nodes(graph, test_X, ind_to_remove)

        true_loss = true_loss / (tot_num_nodes - 1)
        self.log(
            "test_true_loss_soft",
            true_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=self.distributed,
            batch_size=batch_size,
        )

    def predict_step(self, batch, _):
        test_X, _, graph, *_ = batch

        graph = graph[0].unsqueeze(0)  # here we assume that the batch come from the same datset
        batch_size = test_X.shape[0]
        tot_num_nodes = graph.shape[-1]

        ## generate the current topological ordering predicted by the leaf model
        leaf_decreasing_order = []
        for _ in range(tot_num_nodes - 1):
            perm_pred = self.method(test_X)

            ## soft-sum
            cumulate_probs = torch.sigmoid(perm_pred).mean(dim=0)
            ind_leaf = torch.argmax(cumulate_probs, dim=-1).item()

            leaf_decreasing_order.append(ind_leaf)

            # remove the leaf node
            ind_to_remove = torch.Tensor([ind_leaf for k in range(batch_size)]).long()
            graph, test_X = remove_leaf_nodes(graph, test_X, ind_to_remove)

        pred_sig = decreasing_sig_to_perm(leaf_decreasing_order)

        return pred_sig
