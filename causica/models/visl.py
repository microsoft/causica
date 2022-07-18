# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import json
import math
import os
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..datasets.dataset import Dataset
from ..datasets.variables import Variables
from ..models.imodel import IModelForCausalInference
from ..utils.helper_functions import to_tensors
from ..utils.io_utils import save_json
from ..utils.nri_utils import compute_dag_loss, get_feature_indices_per_node, kl_categorical, piecewise_linear
from ..utils.training_objectives import get_input_and_scoring_masks, kl_divergence, negative_log_likelihood
from .pvae_base_model import PVAEBaseModel


class VISL(PVAEBaseModel, IModelForCausalInference):
    """
    Subclass of `models.pvae_base_model.PVAEBaseModel` representing the algorithm VISL (missing value imputation with causal discovery).

    Requires file <data_dir>/<dataset_name>/adj_matrix.csv to evaluate causal discovery against ground truth.
    """

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        gnn_iters: int,
        shared_init_and_final_mappings: bool,
        embedding_dim: int,
        init_prob: float,
        simpler: str = None,
        **_,
    ):
        """
        Args:
            model_id: Unique model ID for referencing this model instance.
            variables: Information about variables/features used by this model.
            save_dir: Location to save any information about this model, including training data.
            device: Device to load model to.
            gnn_iters: Number of message passing iterations for the GNN.
            shared_init_and_final_mappings: Whether all the nodes should use the same MLPs for the initial and final mappings.
            embedding_dim: Dimensionality of the nodes embedding.
            init_prob: Initial probability of having edge.
            simpler: Choose what MLP should be simpler (options are 'forward', 'backward', or None). Specifically,
                'simpler' means to divide by 10 the dimensionality of the hidden layer of the corresponding MLP (with a minimum of 10 units).
        """
        super().__init__(model_id, variables, save_dir, device)

        # Define some useful attributes
        feature_indices_per_node, ordered_nodes = get_feature_indices_per_node(variables)
        with open(os.path.join(self.save_dir, "ordered_nodes.json"), "w", encoding="utf-8") as f:
            json.dump(ordered_nodes, f, indent=4)
        self.num_nodes = len(feature_indices_per_node)
        self.num_edges = self.num_nodes * (self.num_nodes - 1)
        self.input_dim = variables.num_processed_cols

        # Define and initialize Z_edges
        # The learnable parameter is Z_edges_logits. Z_edges is F.softmax(Z_edges_logits, dim=1).
        self.Z_edges_logits = torch.nn.Parameter(
            torch.stack(
                [
                    torch.full([self.num_edges], math.log(1 - init_prob)),
                    torch.full([self.num_edges], math.log(init_prob)),
                ],
                dim=1,
            ).to(device)
        )

        # Define the GNN-based VAE
        self.gnn_vae = GNN_based_VAE(
            embedding_dim=embedding_dim,
            skip_first=True,
            device=device,
            n_iters=gnn_iters,
            num_nodes=self.num_nodes,
            shared_init_and_final_mappings=shared_init_and_final_mappings,
            simpler=simpler,
            feature_indices_per_node=feature_indices_per_node,
        )

        # Create rel_rec and rel_send, which codify the receiving and sending node for each edge
        # Shape of rel_rec and rel_send: [num_edges, num_nodes]
        # The second dimension is a one-hot encoding of the receiver or sender node
        off_diag = np.ones([self.num_nodes, self.num_nodes]) - np.eye(self.num_nodes)
        rel_rec = F.one_hot(torch.tensor(np.where(off_diag)[0], dtype=torch.long))
        rel_send = F.one_hot(torch.tensor(np.where(off_diag)[1], dtype=torch.long))
        self.rel_rec = rel_rec.type(torch.float).to(device)
        self.rel_send = rel_send.type(torch.float).to(device)

        # Define the prior over edge types (favors sparse graphs)
        self.log_prior = torch.log(
            torch.tensor([0.95, 0.05], device=device)
        )  # The no-edge type is the first one (recall the skip_first argument of GNN_based_VAE __init__)

        # Save type of variables. Used in reconstruct method for
        #   1. filling the missing values before applying the GNN-based VAE,
        #   2. processing the output of the GNN-based VAE (i.e. use torch.sigmoid in the binary case)
        types = np.array([v.type_ for v in self.variables._variables])
        if (types == "binary").all():
            self.var_types = "binary"
        elif (types == "continuous").all():
            self.var_types = "continuous"
        elif (types == "categorical").all():
            self.var_types = "categorical"
        else:
            raise ValueError("Right now all the variables need to have the same type")

    def _train(  # type: ignore
        self,
        dataset: Dataset,
        report_progress_callback: Optional[Callable[[str, int, int], None]],
        learning_rate: float,
        batch_size: int,
        epochs: int,
        max_p_train_dropout: float,
        use_dag_loss: bool,
        output_variance: float,
        hard: bool,
        two_steps: bool,
        lambda_nll: float,
        lambda_kl_z: float,
        lambda_kl_A: float,
        lambda_dagloss: float,
        sample_count: int,
    ) -> Dict[str, List[float]]:
        """
        Train the model using the given data.

        Args:
            dataset: Dataset with data and masks in processed form.
            train_output_dir: Path to save any training information to.
            report_progress_callback: Function to report model progress for API.
            learning_rate: Learning rate for Adam optimiser.
            batch_size: Size of minibatches to use.
            epochs: Number of epochs to train for.
            max_p_train_dropout: Maximum fraction of extra training features to drop for each row. 0 is none, 1 is all.
            use_dag_loss: Whether to use the DAG loss regularisation.
            output_variance: The variance for the output of the GNN based VAE.
            hard: Whether to use hard or soft samples for the distribution over edges (if hard=True, the edge weights are just 0/1).
            two_steps: Whether to use the two-step variant of VISL. That is, the first half of training uses only
                the forward MLP and the second half fixes the distribution over edges and only optimizes the forward and backward MLPs.
            lambda_nll: Lambda coefficient for the ELBO term negative-log-likelihood
            lambda_kl_z: Lambda coefficient for the ELBO term lambda*KL(q(z|x) || p(z))
            lambda_kl_A: Lambda coefficient for the ELBO term lambda*KL(q(A) || p(A))
            lambda_dagloss: Lambda coefficient for the dagloss term of the ELBO.
            sample_count: Number of samples to reconstruct.

        Returns:
            train_results (dictionary): training_loss, KL divergence, NLL, dag_loss, training_loss_complete
        """
        # Put PyTorch into train mode.
        self.train()

        # Setting the hard attribute which will be used for training and testing
        self.hard = hard

        # Loading data and mask, creating results_dict
        data, mask = dataset.train_data_and_mask
        results_dict: Dict[str, List] = {
            metric: []
            for metric in [
                "training_loss",
                "kl_z_term",
                "kl_A_term",
                "nll_term",
                "dag_loss_term",
                "training_loss_complete",
            ]
        }

        # Creating the optimizer
        # If two_steps is True, we create a different optimizer for the second half. This optimizer does not optimize over the adjacency matrix.
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        if two_steps:
            named_parameters = list(self.named_parameters())
            all_but_adj_matrix = []
            for a in named_parameters:
                if a[0] != "Z_edges_logits":
                    all_but_adj_matrix.append(a[1])
            optimizer_second_half = torch.optim.Adam(all_but_adj_matrix, lr=learning_rate)

        # Creating the dataloader
        tensor_dataset = TensorDataset(*to_tensors(data, mask, device=self.device))
        dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

        # If DAG loss is used, it appears after 'epochs_without_dagloss' epochs and its coefficient (lambda) grows linearly
        # during 10% of the total number of epochs until 1. This scheme is used for lambda because of empirical
        # reasons (DAG loss might take over the training if it is used with coefficient 1.0 from the beginning).
        if use_dag_loss:
            epochs_without_dagloss = 5
            if (epochs_without_dagloss + 0.1 * epochs) > epochs:
                warnings.warn("max lambda will not be achieved")

        best_train_loss = np.nan
        for epoch in range(epochs):
            training_loss_full = 0.0
            nll_term_full = 0.0
            kl_z_term_full = 0.0
            kl_A_term_full = 0.0
            dag_loss_term_full = 0.0
            training_loss_complete_full = 0.0

            # Set the optimizer_to_use depending on whether we are using the two-steps variant or not.
            if not two_steps:
                optimizer_to_use = optimizer
                only_forward = False
            elif epoch < epochs // 2:
                optimizer_to_use = optimizer
                only_forward = True
            else:
                optimizer_to_use = optimizer_second_half
                only_forward = False

            for x, mask_train_batch in dataloader:
                # Drop additional values (same procedure as PVAE)
                input_mask, scoring_mask = get_input_and_scoring_masks(
                    mask_train_batch,
                    max_p_train_dropout=max_p_train_dropout,
                    score_imputation=True,
                    score_reconstruction=True,
                )

                # Apply the GNN-based VAE
                (x_reconstructed, _), _, encoder_output = self.reconstruct(
                    x, input_mask, only_forward=only_forward, count=sample_count
                )

                # Loss: lambda_nll * NLL +
                #       lambda_kl_z * KL(q(z)||p(z)) +
                #       lambda_kl_A * KL(q(A)||p(A)) +
                #       piecewise_linear_temperature * lambda_dagloss * dag_loss
                # NLL
                dec_logvar = torch.full_like(x_reconstructed, math.log(output_variance))
                categorical_lik_coeff = 1.0
                nll = negative_log_likelihood(
                    x, x_reconstructed, dec_logvar, self.variables, categorical_lik_coeff, scoring_mask
                )
                nll_term = lambda_nll * nll
                # KL(q(z)||p(z)) term
                kl_z_term = lambda_kl_z * kl_divergence(encoder_output).sum()
                # KL(q(A)||p(A)) term
                probs = F.softmax(self.Z_edges_logits, dim=-1)  # [num_edges, num_edge_types]
                scale_batch_size = x.shape[0] / data.shape[0]
                kl_A_term = (
                    lambda_kl_A
                    * scale_batch_size
                    * kl_categorical(probs[None, :, :], self.log_prior[None, None, :], self.num_nodes)
                )  # The input to kl_categorical is [num_sims,num_edges,num_edge_types] (from original NRI)
                # Loss
                loss = nll_term + kl_z_term + kl_A_term

                if use_dag_loss:
                    assert probs.shape[1] == 2  # Checks that we have 2 edge types (no edge and edge)
                    dag_loss = compute_dag_loss(probs[:, 1], self.num_nodes)
                    dag_loss_term = scale_batch_size * lambda_dagloss * dag_loss
                    loss += (
                        piecewise_linear(epoch, start=epochs_without_dagloss, width=0.1 * epochs)
                        * scale_batch_size
                        * lambda_dagloss
                        * dag_loss
                    )
                    loss_complete = loss + dag_loss_term
                else:
                    dag_loss_term = torch.tensor(np.nan)
                    loss_complete = loss

                # Optimize
                optimizer_to_use.zero_grad()
                loss.backward()
                optimizer_to_use.step()

                # Adding loss to the total (and nll/kl/dag_loss)
                training_loss_full += loss.item()
                nll_term_full += nll_term.item()
                kl_z_term_full += kl_z_term.item()
                kl_A_term_full += kl_A_term.item()
                dag_loss_term_full += dag_loss_term.item()
                training_loss_complete_full += loss_complete.item()

            # Save model if the loss has improved
            if np.isnan(best_train_loss) or training_loss_complete_full < best_train_loss:
                best_train_loss = training_loss_complete_full
                best_kl_z_term = kl_z_term_full
                best_kl_A_term = kl_A_term_full
                best_nll_term = nll_term_full
                best_dag_loss_term = dag_loss_term_full
                best_epoch = epoch
                self.save()

            # Save useful quantities.
            # Training_loss is the loss that is minimized.
            # Training_loss_complete is the loss that is printed and based on which the saving is made.
            # Both can differ if there is some temperature growing as the training evolves (as is the case of the piecewise linear temperature for dag_loss)
            results_dict["training_loss"].append(training_loss_full)
            results_dict["nll_term"].append(nll_term_full)
            results_dict["kl_z_term"].append(kl_z_term_full)
            results_dict["kl_A_term"].append(kl_A_term_full)
            results_dict["dag_loss_term"].append(dag_loss_term_full)
            results_dict["training_loss_complete"].append(training_loss_complete_full)

            if np.isnan(training_loss_full):
                print("Training loss is NaN. Exiting early.", flush=True)
                break

            if report_progress_callback:
                report_progress_callback(self.model_id, epoch + 1, epochs)

            print(
                f"Epoch: {epoch} train_loss (complete): {training_loss_complete_full:.2f} nll_term: {nll_term_full:.2f}"
                f"kl_z_term: {kl_z_term_full:.2f} kl_A_term: {kl_A_term_full:.2f} dl_term: {dag_loss_term_full:.2f}",
                flush=True,
            )

        print(
            f"Best model found at epoch {best_epoch}, with train_loss {best_train_loss:.2f}, kl_z_term {best_kl_z_term:.2f},"
            f"kl_A_term {best_kl_A_term:.2f}, nll_term {best_nll_term:.2f} and dl {best_dag_loss_term:.2f}",
            flush=True,
        )

        # Saving the predicted adjacency matrix
        adj_matrix_predicted = self.get_adj_matrix()
        np.savetxt(
            os.path.join(self.save_dir, "adj_matrix_predicted.csv"), adj_matrix_predicted, delimiter=",", fmt="%.8f"
        )
        np.savetxt(
            os.path.join(self.save_dir, "adj_matrix_predicted_round.csv"),
            adj_matrix_predicted.round(),
            delimiter=",",
            fmt="%i",
        )

        # Saving the values of the losses obtained during training
        save_json(results_dict, os.path.join(self.save_dir, "training_losses.json"))

    @classmethod
    def name(cls) -> str:
        return "visl"

    def get_adj_matrix(self):
        """
        Returns the adjacency matrix as a numpy array.
        """
        # This is currently implemented for the case when there are only two edge types (edge and no-edge)
        assert self.Z_edges_logits.shape[1] == 2
        Z_edge_logits = self.Z_edges_logits.detach().cpu().numpy()  # [num_edges, 2]
        prob = np.exp(Z_edge_logits) / np.sum(np.exp(Z_edge_logits), axis=-1, keepdims=True)  # [num_edges, 2]
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes))
        mask = np.ones((self.num_nodes, self.num_nodes), dtype=bool) & ~np.eye(self.num_nodes, dtype=bool)
        adj_matrix[mask] = prob[:, 1]
        return adj_matrix

    def encode(self, data: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This is included because it is an abstract method of the parent class IModelForObjective, but it is not used in the implementation.
        All the encoding and decoding is performed by the GNN_based_VAE
        """
        raise NotImplementedError()

    def reconstruct(
        self,
        data: torch.Tensor,
        mask: Optional[torch.Tensor],
        sample: bool = True,
        count: int = 1,
        only_forward: bool = False,
        **kwargs,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Reconstruct data by filling missing values and passing them through the GNN-based VAE.

        Args:
            data: Input data with shape (batch_size, input_dim).
            mask: Mask indicating observed variables with shape (batch_size, input_dim). 1 is observed, 0 is un-observed.
            count: Number of samples to reconstruct.
            only_forward: Whether to use only the forward MLP in the message passing (this is used in the two-steps variant).

        Returns:
            data_reconstructed: reconstructed data, with shape (count, batch_size, input_dim). count is removed if 1.
            (encoder_mean, encoder_logvar): Output of the encoder. Both are shape (batch_size, latent_dim)
        """

        # Filling non-available values before applying the GNN-based VAE
        assert mask is not None
        data_zi = data.clone()
        assert ((mask == 0.0) | (mask == 1.0)).all()
        if self.var_types == "continuous":
            data_zi[mask == 0.0] = 0.0
        elif self.var_types == "binary":
            data_zi[mask == 0.0] = 0.5
        elif self.var_types == "categorical":
            raise ValueError("Categorical data type not supported yet")

        # Apply the GNN-based VAE
        edges = torch.stack(
            [F.gumbel_softmax(self.Z_edges_logits, tau=0.5, hard=self.hard) for _ in range(count)], dim=0
        )  # [count, num_edges, num_edge_types]
        data_reconstructed, encoder_mean, encoder_logvar = self.gnn_vae(
            data_zi, edges, self.rel_rec, self.rel_send, only_forward=only_forward, sample_count=count
        )  # [batch_size, D]

        # Using sigmoid if data is binary (the nll function assumes that the data is in [0,1] for binary variables)
        if self.var_types == "binary":
            data_reconstructed = torch.sigmoid(data_reconstructed)
        return (data_reconstructed, None), None, (encoder_mean, encoder_logvar)  # type: ignore


class GNN_based_VAE(nn.Module):
    """
    GNN-based VAE that
        1. encodes the variables into the initial embedding for each node,
        2. does the GNN message passing,
        3. decodes the final embedding for each node into the variable value.
    """

    def __init__(
        self,
        embedding_dim: int,
        device: torch.device,
        skip_first: bool,
        n_iters: int,
        num_nodes: int,
        shared_init_and_final_mappings: bool,
        feature_indices_per_node: list,
        simpler: str = None,
    ):
        """
        Args
            embedding_dim: Dimensionality of each node embedding.
            device: Device to load model to.
            skip_first: Whether to use a no-edge type.
            n_iters: Number of GNN message passing iterations.
            num_nodes: Number of nodes.
            shared_init_and_final_mappings: Whether all the nodes should use the same MLPs for the initial and final mappings.
            simpler: Choose what MLP should be simpler (options are 'forward', 'backward', or None). Specifically,
                'simpler' means to divide by 10 the dimensionality of the hidden layer of the corresponding MLP (with a minimum of 10 units).
            feature_indices_per_node: the i-th element of this list is a list containing the indexes that correspond to the i-th node.
        """
        super().__init__()
        self.device = device
        self.n_iters = n_iters
        self.num_nodes = num_nodes
        self.shared_init_and_final_mappings = shared_init_and_final_mappings
        self.feature_indices_per_node = feature_indices_per_node

        # Initial mapping (input_dim -> embedding_dim -> embedding_dim)
        if shared_init_and_final_mappings:
            input_dim = len(feature_indices_per_node[0])
            assert all(
                len(feature_indices) == input_dim for feature_indices in feature_indices_per_node
            ), "The encoding MLP cannot be shared between nodes"
            self.init_map_fc1 = nn.Linear(input_dim, embedding_dim)
            self.init_map_fc2 = nn.Linear(embedding_dim, 2 * embedding_dim)
        else:
            self.init_map_fc1_list = nn.ModuleList(
                [nn.Linear(len(fi), embedding_dim) for fi in feature_indices_per_node]
            )
            self.init_map_fc2_list = nn.ModuleList(
                [nn.Linear(embedding_dim, 2 * embedding_dim) for fi in feature_indices_per_node]
            )

        # Determining the dimensionality of the hidden layer depending on the 'simpler' argument
        if simpler is None:
            forward_hidden_dim = embedding_dim
            backward_hidden_dim = embedding_dim
        elif simpler == "forward":
            forward_hidden_dim = max(embedding_dim // 10, 10)
            backward_hidden_dim = embedding_dim
        elif simpler == "backward":
            forward_hidden_dim = embedding_dim
            backward_hidden_dim = max(embedding_dim // 10, 10)
        else:
            raise ValueError("The argument 'simpler' must be None, forward or backward")

        # Node-to-edge MLPs (2*embedding_dim -> embedding_dim -> embedding_dim).
        # There are 2*K MLPs: K for the forward message passing and K for the backward (K is the number of edge types)
        num_edge_types = 2
        self.n2e_fc1_forward = nn.ModuleList(
            [nn.Linear(2 * embedding_dim, forward_hidden_dim) for _ in range(num_edge_types)]
        )
        self.n2e_fc2_forward = nn.ModuleList(
            [nn.Linear(forward_hidden_dim, embedding_dim) for _ in range(num_edge_types)]
        )
        self.n2e_fc1_backward = nn.ModuleList(
            [nn.Linear(2 * embedding_dim, backward_hidden_dim) for _ in range(num_edge_types)]
        )
        self.n2e_fc2_backward = nn.ModuleList(
            [nn.Linear(backward_hidden_dim, embedding_dim) for _ in range(num_edge_types)]
        )

        # Edge-to-node MLP (embedding_dim -> embedding_dim -> embedding_dim)
        self.e2n_fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.e2n_fc2 = nn.Linear(embedding_dim, embedding_dim)

        # Final mapping (embedding_dim -> embedding_dim -> input_dim)
        if shared_init_and_final_mappings:
            self.final_map_fc1 = nn.Linear(embedding_dim, embedding_dim)
            self.final_map_fc2 = nn.Linear(embedding_dim, input_dim)
        else:
            self.final_map_fc1_list = nn.ModuleList(
                [nn.Linear(embedding_dim, embedding_dim) for fi in feature_indices_per_node]
            )
            self.final_map_fc2_list = nn.ModuleList(
                [nn.Linear(embedding_dim, len(fi)) for fi in feature_indices_per_node]
            )

        self.skip_first_edge_type = skip_first

        self.to(device)

    def forward(
        self,
        inp: torch.Tensor,
        edges_weights: torch.Tensor,
        rel_rec: torch.Tensor,
        rel_send: torch.Tensor,
        only_forward: bool,
        sample_count: int,
    ):
        """
        Forward pass of the GNN-based VAE. This includes:
            1. encoding the variables into the initial embedding for each node,
            2. doing the GNN message passing,
            3. decoding the final embedding for each node into the variable value.

        Args:
            inp: The input to the GNN-based VAE. Shape (batch_size, num_features).
            edges_weights: The edge weights used for the message passing. Add up to 1 in the last dim. Shape (sample_count, num_edges, num_edge_types).
            rel_rec: Tensor identifying the receiving node for each edge. The second dimension is a one-hot encoding of the receiver node. Shape (num_edges, num_nodes).
            rel_send: Tensor identifying the sending node for each edge. The second dimension is a one-hot encoding of the sending node. Shape (num_edges, num_nodes).
            only_forward: Whether to use only the forward MLP for the message passing.
            sample_count: Number of samples to reconstruct.

        Return:
            output: The reconstructed data. Shape (sample_count, batch_size, num_features). sample_count is removed if 1.
            encoder_mean, encoder_logvar: Output of the encoder. Both are shape (batch_size, latent_dim)
        """
        # Encoding the variables. The output is node_embedding_mean and node_embedding_logvar
        if self.shared_init_and_final_mappings:
            node_embedding = F.relu(self.init_map_fc1(inp[:, torch.tensor(self.feature_indices_per_node)]))
            node_embedding = self.init_map_fc2(node_embedding)  # [batch_size, num_nodes, 2*embedding_dim]
            node_embedding_mean, node_embedding_logvar = node_embedding.chunk(
                2, dim=-1
            )  # Each [batch_size, num_nodes, embedding_dim]
        else:
            node_embedding_mean_list, node_embedding_logvar_list = [], []
            for i in range(self.num_nodes):
                node_embedding_i = F.relu(
                    self.init_map_fc1_list[i](inp[:, self.feature_indices_per_node[i]])
                )  # [batch_size, embedding_dim]
                node_embedding_i = self.init_map_fc2_list[i](node_embedding_i)  # [batch_size, 2*embedding_dim]
                node_embedding_i_mean, node_embedding_i_logvar = node_embedding_i.chunk(
                    2, dim=-1
                )  # each [batch_size, embedding_dim]
                node_embedding_mean_list.append(node_embedding_i_mean)
                node_embedding_logvar_list.append(node_embedding_i_logvar)
            node_embedding_mean = torch.stack(node_embedding_mean_list, dim=1)  # [batch_size, num_nodes, embedding_dim]
            node_embedding_logvar = torch.stack(
                node_embedding_logvar_list, dim=1
            )  # [batch_size, num_nodes, embedding_dim]

        # Sampling the latent variables
        # Clamp node_embedding_logvar (better numerical stability)
        node_embedding_logvar = torch.clamp(node_embedding_logvar, -20, 20)
        node_embedding_stddev = torch.sqrt(torch.exp(node_embedding_logvar))
        gaussian = tdist.Normal(node_embedding_mean, node_embedding_stddev)
        node_embedding_samples = gaussian.rsample((sample_count,)).to(
            self.device
        )  # Shape [sample_count, batch_size, num_nodes, embedding_dim]

        # Doing the GNN message passing
        node_embedding = node_embedding_samples  # Shape [sample_count, batch_size, num_nodes, embedding_dim]
        start_idx = 1 if self.skip_first_edge_type else 0
        for i in range(self.n_iters):
            # Node-to-edge
            edge_embedding = torch.cat(
                [torch.matmul(rel_send, node_embedding), torch.matmul(rel_rec, node_embedding)], dim=-1
            )  # [sample_count, batch_size, num_edges, 2*embedding_dim]
            edge_embedding_forward_total = torch.zeros(
                edge_embedding.size(0),
                edge_embedding.size(1),
                edge_embedding.size(2),
                self.n2e_fc2_forward[0].out_features,
                device=self.device,
            )  # shape: [sample_count, batch_size, num_edges, embedding_dim]
            edge_embedding_backward_total = torch.zeros(
                edge_embedding.size(0),
                edge_embedding.size(1),
                edge_embedding.size(2),
                self.n2e_fc2_backward[0].out_features,
                device=self.device,
            )  # shape: [sample_count, batch_size, num_edges, embedding_dim]
            # Using one MLP for each edge type. If there is no-edge then the first MLP is skipped.
            for k in range(start_idx, len(self.n2e_fc1_forward)):
                edge_embedding_forward_total += edges_weights[:, :, k][:, None, :, None] * F.relu(
                    self.n2e_fc2_forward[k](
                        F.relu(self.n2e_fc1_forward[k](edge_embedding))
                    )  # shape: [sample_count, batch_size, num_edges, embedding_dim]
                )
                if not only_forward:
                    edge_embedding_backward_total += edges_weights[:, :, k][:, None, :, None] * F.relu(
                        self.n2e_fc2_backward[k](
                            F.relu(self.n2e_fc1_backward[k](edge_embedding))
                        )  # shape: [sample_count, batch_size, num_edges, embedding_dim]
                    )
            # Edge-to-node
            if only_forward:
                node_embedding = torch.matmul(
                    rel_rec.transpose(0, 1), edge_embedding_forward_total
                )  # [sample_count, batch_size, num_nodes, embedding_dim]
            else:
                node_embedding = torch.matmul(rel_rec.transpose(0, 1), edge_embedding_forward_total) + torch.matmul(
                    rel_send.transpose(0, 1), edge_embedding_backward_total
                )  # [sample_count, batch_size, num_nodes, embedding_dim]
            node_embedding = F.relu(self.e2n_fc1(node_embedding))
            node_embedding = F.relu(
                self.e2n_fc2(node_embedding)
            )  # [sample_count, batch_size, num_nodes, embedding_dim]

        # Decoding the final embedding for each node into the variable value.
        output_final = torch.zeros(
            node_embedding.shape[0], node_embedding.shape[1], inp.shape[1], device=self.device
        )  # [sample_count, batch_size, num_features]
        if self.shared_init_and_final_mappings:
            output = F.relu(self.final_map_fc1(node_embedding))  # [sample_count, batch_size, num_nodes, embedding_dim]
            output = self.final_map_fc2(output)  # [sample_count, batch_size, num_nodes, input_dim]
            output_final[:, :, torch.tensor(self.feature_indices_per_node)] = output
        else:
            for i in range(self.num_nodes):
                output_i = F.relu(
                    self.final_map_fc1_list[i](node_embedding[:, :, i, :])
                )  # [sample_count, batch_size, embedding_dim]
                output_final[:, :, self.feature_indices_per_node[i]] = self.final_map_fc2_list[i](
                    output_i
                )  # [sample_count, batch_size, len(feature_indices_per_node[i])]

        if sample_count == 1:
            output_final = output_final[0, :, :]
        return (
            output_final,
            node_embedding_mean.reshape(node_embedding_mean.shape[0], -1),
            node_embedding_logvar.reshape(node_embedding_logvar.shape[0], -1),
        )
