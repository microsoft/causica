from __future__ import annotations

import typing
from typing import Any, Optional, Union, cast

import numpy as np
import torch

from ...datasets.dataset import Dataset
from ...datasets.variables import Variable, Variables
from ...models.deci.base_distributions import GaussianBase
from ...models.deci.variational_distributions import VarDistA_ENCO, VarDistA_ENCO_ADMG
from ...nn_modules.activations import NormalDistribution
from ...utils.causality_utils import dag2admg, intervention_to_tensor
from ...utils.helper_functions import to_tensors
from ...utils.nri_utils import edge_prediction_metrics_multisample
from ...utils.torch_utils import generate_fully_connected
from .deci import DECI


def _separate_latent_and_observed_variables(
    variables: Variables,
) -> tuple[Variables, Variables, Variables]:
    """Restructures the variables into latent variables and observed variables.

    Args:
        variables: Set of all variables with latent variables having the group name "latent".

    Returns:
        Tuple of all variables, observed variables and latent variables.
    """
    # Separate out the latent variables to modify the group name.
    variables_list = variables.as_list()
    assert len([v for v in variables_list if v.is_latent]) > 0, "No latent variables were found."

    observed_variables_list = [v for v in variables_list if not v.is_latent]
    latent_variables_list = [v for v in variables_list if v.is_latent]

    variables = Variables(observed_variables_list + latent_variables_list)
    latent_variables = Variables(latent_variables_list)
    observed_variables = Variables(observed_variables_list)

    return variables, observed_variables, latent_variables


def _add_latent_variables(variables: Variables, num_latent_variables: int) -> Variables:
    """Adds latent variables to the given variables.

    The observed variables appear first in the list of variables, followed by the latent variables. This convention is
    used throughout the DDECI implementation and is crucial for everything to work correctly.
    """
    latent_variables = [
        Variable(name=f"U{i}", query=True, type="continuous", lower=0.0, upper=0.0, is_latent=True)
        for i in range(num_latent_variables)
    ]

    return Variables(variables.as_list() + latent_variables)


class DDECI(DECI):
    """Deconfounding DECI: an extension of DECI that accounts for the presence of latent confounders.

    This model uses a DAG parameterisation over both the observed and latent variables. The number of latent variables
    must be specified in advance, and no restrictions are placed on the latent variables being confounders between pairs
    of variables.
    """

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        imputer_layer_sizes: Optional[list[int]] = None,
        inference_network_layer_sizes: Optional[list[int]] = None,
        **kwargs,
    ):
        """
        Args:
            variables: Information about both the observed variables and the latent variables.
            inference_network_layer_sizes: The layer sizes of the inference network used to obtain q(u | x).
            **kwargs: Additional keyword arguments to be passed to DECI.__init__().
        """
        variables, observed_variables, latent_variables = _separate_latent_and_observed_variables(variables)
        super().__init__(model_id, variables, save_dir, device, **kwargs)

        self.observed_variables = observed_variables
        self.observed_dim_all = self.observed_variables.num_processed_non_aux_cols
        self.observed_num_nodes = self.observed_variables.num_groups

        self.latent_variables = latent_variables
        self.latent_dim_all = self.latent_variables.num_processed_non_aux_cols
        self.latent_num_nodes = self.latent_variables.num_groups

        # Construct inference network.
        inference_network_layer_sizes = inference_network_layer_sizes or [max(80, 2 * self.observed_dim_all)] * 2
        self.inference_network = generate_fully_connected(
            input_dim=self.observed_dim_all,
            output_dim=2 * self.latent_dim_all,
            hidden_dims=inference_network_layer_sizes,
            non_linearity=torch.nn.LeakyReLU,
            activation=NormalDistribution,
            device=self.device,
            normalization=self.norm_layer,
        )

        # (Re-)construct imputer network.
        if self.imputation:
            imputation_input_dim = 2 * self.observed_dim_all
            imputation_output_dim = 2 * self.observed_dim_all
            imputer_layer_sizes = imputer_layer_sizes or [max(80, imputation_input_dim)] * 2
            self.imputer_network = generate_fully_connected(
                input_dim=imputation_input_dim,
                output_dim=imputation_output_dim,
                hidden_dims=imputer_layer_sizes,
                non_linearity=torch.nn.LeakyReLU,
                activation=NormalDistribution,
                device=self.device,
                normalization=self.norm_layer,
                res_connection=self.res_connection,
            )

        # Likelihoods for the latent variables.
        self.latent_variable_likelihood = GaussianBase(
            self.latent_dim_all, device=self.device, train_base=False, log_scale_init=0
        )

    @classmethod
    def name(cls) -> str:
        return "ddeci"

    def reconstruct_x(
        self,
        x: torch.Tensor,
        most_likely_graph: bool = False,
        max_batch_size: int = 1024,
    ) -> torch.Tensor:
        """Reconstructs the observations using a sample from q(u | x) and q(A).

        This function is mainly used for the purpose of visualisation.

        Args:
            x: Observations.
            most_likely_graph: Indicates whether to deterministically pick the most probable graph.
            max_batch_size: Maximum batch size to use for DECI when drawing samples.

        Returns:
            Reconstruction of the observations.
        """
        W_adj = self.get_weighted_adj_matrix(
            do_round=most_likely_graph,
            samples=1,
            most_likely_graph=most_likely_graph,
        ).squeeze(0)

        # Sample exogenous noise.
        z = self._sample_base(x.shape[0]).detach()

        # Sample from q(u | x).
        qu_x = self.inference_network(x)
        u = qu_x.sample()

        # Treat latent variables as interventions as we've already sampled them. Since latent variables are source
        # nodes, this is the same as conditioning on them.
        intervention_idxs: Optional[torch.Tensor] = torch.arange(
            self.observed_num_nodes, self.num_nodes, device=self.device
        )

        # Observations to feed into the likelihood.
        x_rec_list = []

        for u_batch, z_batch in zip(
            torch.split(u, max_batch_size, dim=0),
            torch.split(z, max_batch_size, dim=0),
        ):
            (intervention_idxs, intervention_mask, intervention_values,) = intervention_to_tensor(
                intervention_idxs,
                u_batch,
                self.variables.group_mask,
                device=self.device,
            )
            x_rec = self.ICGNN.simulate_SEM(z_batch, W_adj, intervention_mask, intervention_values)
            x_rec_list.append(x_rec[:, : -self.latent_dim_all])

        x_rec = torch.cat(x_rec_list, dim=0)
        assert x_rec.shape[0] == x.shape[0]
        return x_rec

    def _log_prob(
        self, x: torch.Tensor, predict: torch.Tensor, intervention_mask: Optional[torch.Tensor] = None, **_
    ) -> torch.Tensor:
        assert x.shape[-1] == self.processed_dim_all, "Incorrect variable dimensions."
        assert predict.shape == x.shape, "Mismatch between dimensions."

        if intervention_mask is None:
            intervention_mask = torch.zeros(self.processed_dim_all, device=self.device, dtype=torch.bool)

        log_prob_x, log_prob_u = self._log_prob_all(x, predict, intervention_mask)

        return log_prob_x + log_prob_u

    def _log_prob_all(
        self,
        x: torch.Tensor,
        predict: torch.Tensor,
        intervention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the log probability of the observed data and latent variables given the predictions from the SEM.

        Args:
            x: Array of size (processed_dim_all) or (batch_size, processed_dim_all).
            predict: Array of size the same shape as x.
            intervention_mask: Array containing indicators of variables that have been intervened.
                These will not be considered for log probability computation. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor] containing the log probability of the non-intervened observations and
                latent variables.
        """

        assert (
            len(intervention_mask) == self.processed_dim_all
        ), "intervention_mask is not compatible with the number of variables."
        intervention_mask_x = intervention_mask.clone()
        intervention_mask_x[-self.latent_dim_all :] = True
        intervention_mask_u = intervention_mask[-self.latent_dim_all :]

        u = x[..., -self.latent_dim_all :]
        u_predict = predict[..., -self.latent_dim_all :]

        log_prob_x = super()._log_prob(x, predict, intervention_mask_x)
        log_prob_u = self.latent_variable_likelihood.log_prob(u - u_predict)

        if intervention_mask_u is not None:
            log_prob_u[..., intervention_mask_u] = 0.0

        log_prob_u = log_prob_u.sum(-1)

        return log_prob_x, log_prob_u

    @typing.no_type_check
    def log_prob(
        self,
        X: Union[torch.Tensor, np.ndarray],
        intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        intervention_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        Nsamples_per_graph: int = 1,
        Ngraphs: Optional[int] = 1,
        most_likely_graph: bool = False,
        most_likely_u: bool = False,
    ) -> np.ndarray:
        """Evaluate the log-probability of observations X.

        Evaluate log-probability of observations X. Optionally this evaluation can be subject to an intervention on our
        causal model. Then, the log probability of non-intervened variables is computed.

        Args:
            X: torch.Tensor of shape (batch_size, input_dim) containing the observations we want to evaluate
            Nsamples: int containing number of graph samples to draw.
            most_likely_graph: bool indicatng whether to deterministically pick the most probable graph under the
                approximate posterior instead of sampling graphs.
            most_likely_u: bool indicating whether to use to mean of q(u | x) instead of drawing samples.
            intervention_idxs: torch.Tensor of shape (input_dim) optional array containing indices of variables that
                have been intervened.
            intervention_values: torch.Tensor of shape (input_dim) optional array containing values for variables that
                have been intervened.
            Nsamples_per_graph: int containing number of samples to draw
            Ngraphs: Number of different graphs to sample for graph posterior marginalisation. If None, defaults to Nsamples

        Returns:
            log_prob: torch.tensor  (batch_size)
        """
        (X,) = to_tensors(X, device=self.device, dtype=torch.float)

        log_prob_samples = []
        for _ in range(Nsamples_per_graph):

            # Sample from q(u | x).
            qu_x = self.inference_network(X)
            if most_likely_u:
                u = qu_x.loc
            else:
                u = qu_x.sample()

            # Add latent variables to the intervention indices.
            intervention_idxs_all, intervention_values_all = self.add_latents_to_intervention(
                u, intervention_idxs, intervention_values
            )

            x_all = torch.cat([X, u], dim=-1)

            log_prob_sample = super().log_prob(
                x_all,
                most_likely_graph=most_likely_graph,
                intervention_idxs=intervention_idxs_all,
                intervention_values=intervention_values_all,
                Nsamples_per_graph=1,
                Ngraphs=1,
            )
            log_prob_samples.append(torch.as_tensor(log_prob_sample, device=self.device))

        log_prob = torch.logsumexp(torch.stack(log_prob_samples, dim=0), dim=0) - np.log(Nsamples_per_graph)
        return log_prob.detach().cpu().numpy().astype(np.float64)

    def _counterfactual(
        self,
        X: Union[torch.Tensor, np.ndarray],
        W_adj: torch.Tensor,
        intervention_idxs: Union[torch.Tensor, np.ndarray] = None,
        intervention_values: Union[torch.Tensor, np.ndarray] = None,
    ) -> torch.Tensor:
        """Calculates a counterfactual for a given input X and given graph A_sample.

        Generates the counterfactual by first sampling u from q(u | x), then computing the exogenous noise terms. Takes
        a single sample of u per counterfactual.

        Args:
            X: torch.Tensor of shape (Nsamples, input_dim) containing the observations we want to evaluate
            W_adj: torch.Tensor of shape (input_dim, input_dim) containing the weighted adjacency matrix of the graph
            intervention_idxs: torch.Tensor of shape (input_dim) optional array containing indices of variables that have been intervened.
            intervention_values: torch.Tensor of shape (input_dim) optional array containing values for variables that have been intervened.

        Returns:
            counterfactual: torch.Tensor of shape (Nsamples, input_dim) containing the counterfactuals
        """

        (X,) = to_tensors(X, device=self.device, dtype=torch.float)

        gumbel_max_regions = self.variables.processed_cols_by_type["categorical"]
        gt_zero_region = [j for i in self.variables.processed_cols_by_type["binary"] for j in i]

        with torch.no_grad():
            # Sample u from q(u | x).
            qu_x = self.inference_network(X)
            u_sample = qu_x.sample()
            x_all = torch.cat([X, u_sample], dim=-1)

            # Add sampled latent variables to intervention values and idxs.
            # intervention_idxs_all and intervention_values_all include the indices and values the latent variables.
            intervention_idxs_all, intervention_values_all = self.add_latents_to_intervention(
                u_sample, intervention_idxs, intervention_values
            )
            (_, intervention_mask_all_tensor, intervention_values_all_tensor,) = intervention_to_tensor(
                intervention_idxs_all,
                intervention_values_all,
                self.variables.group_mask,
                device=self.device,
            )

            # Infer exogeneous noise variables - this currently only supports continuous variables.
            predict = self.ICGNN.predict(x_all, W_adj)
            noise_variable_posterior_samples = torch.zeros_like(x_all)

            typed_regions = self.variables.processed_cols_by_type
            # Continuous
            continuous_range = [i for region in typed_regions["continuous"] for i in region]
            if continuous_range:
                # Additive Noise Model applies
                noise_variable_posterior_samples[..., continuous_range] = (
                    x_all[..., continuous_range] - predict[..., continuous_range]
                )

            # Categorical
            if "categorical" in typed_regions:
                for region, likelihood in zip(typed_regions["categorical"], self.likelihoods["categorical"]):
                    noise_variable_posterior_samples[..., region] = likelihood.posterior(
                        x_all[..., region], predict[..., region]
                    )

            # Binary: this operation can be done in parallel, so no loop over individual variables
            binary_range = [i for region in typed_regions["binary"] for i in region]
            if binary_range:
                noise_variable_posterior_samples[..., binary_range] = self.likelihoods["binary"].posterior(
                    x_all[..., binary_range], predict[..., binary_range]
                )

            # Get counterfactual by intervening on the graph and forward propagating using the inferred noise variable
            X_cf_all = self.ICGNN.simulate_SEM(
                noise_variable_posterior_samples,
                W_adj,
                intervention_mask_all_tensor,
                intervention_values_all_tensor,
                gumbel_max_regions,
                gt_zero_region,
            )

            return X_cf_all[..., : -self.latent_dim_all]

    def sample(self, *args, **kwargs):
        samples = super().sample(*args, **kwargs)
        return samples[..., : -self.latent_dim_all]

    def sample_all(
        self,
        Nsamples: int = 100,
        most_likely_graph: bool = False,
        intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        intervention_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        samples_per_graph: int = 1,
        max_batch_size: int = 1024,
    ) -> torch.Tensor:
        """Samples both the latent and observed variables from the causal flow model.

        Args:
            Nsamples: int containing number of samples to draw
            most_likely_graph: bool indicatng whether to deterministically pick the most probable graph under the approximate posterior or to draw a new graph for every sample
            intervention_idxs: torch.Tensor of shape (input_dim) optional array containing indices of variables that have been intervened.
            intervention_values: torch.Tensor of shape (input_dim) optional array containing values for variables that have been intervened.
            samples_per_graph: how many samples to draw per graph sampled from the posterior. If most_likely_graph is true, this variable will default to 1,
            max_batch_size: maximum batch size to use for DECI when drawing samples. Larger is faster but more memory intensive
        Returns:
            samples: torch.Tensor of shape (Nsamples, input_dim).
        """
        return super().sample(
            Nsamples, most_likely_graph, intervention_idxs, intervention_values, samples_per_graph, max_batch_size
        )

    def add_latents_to_intervention(
        self,
        u: torch.Tensor,
        intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        intervention_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adds the indices and values of the latent variables to the intervention variables."""
        if intervention_idxs is not None:
            intervention_idxs_all = torch.cat(
                [
                    torch.as_tensor(intervention_idxs, device=self.device),
                    torch.arange(self.observed_num_nodes, self.num_nodes, device=self.device),
                ]
            )
            intervention_values_all = torch.cat(
                [torch.as_tensor(intervention_values, device=self.device).repeat(u.shape[0], 1), u], dim=-1
            )
        else:
            intervention_idxs_all = torch.arange(self.observed_num_nodes, self.num_nodes, device=self.device)
            intervention_values_all = u

        return intervention_idxs_all, intervention_values_all

    def get_params_variational_distribution(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Given a batch of samples x with missing values, returns the mean and scale of variational
        approximation over missing values.
        """
        if not self.all_cts:
            raise NotImplementedError("Imputation is only implemented for all-continuous data.")
        x_obs = x * mask
        input_net = torch.cat([x_obs, mask], dim=1)  # Shape (batch_size, 2 * processed_dim_all)
        qx = self.imputer_network(input_net)
        return qx.loc, qx.scale

    def _ELBO_terms(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        """Computes all terms involved in the ELBO.

        This method is re-implemented to avoid unecessary back-propagation computations.

        Args:
            X: batched samples from the dataset, size (batch_size, input_dim).

        Returns:
            dict[key, torch.Tensor] containing all the terms involved in the ELBO.
        """
        # Get adjacency matrix with weights
        A_sample = self.get_adj_matrix_tensor(do_round=False, samples=1, most_likely_graph=False).squeeze(0)
        if self.mode_adjacency == "learn":
            factor_q = 1.0
        elif self.mode_adjacency in ["upper", "lower"]:
            factor_q = 0.0
        else:
            raise NotImplementedError(f"Adjacency mode {self.mode_adjacency} is not implemented")

        # Get weighted adjacency matrix.
        W_adj = A_sample * self.ICGNN.get_weighted_adjacency()

        # Get q(u | x), draw samples and compute entropy.
        qu_x = self.inference_network(X)
        U = qu_x.rsample()
        log_qu_x = -qu_x.entropy().sum(-1)

        # Concatenate x and u and obtain log-likelihoods.
        x_all = torch.cat([X, U], dim=-1)
        predict = self.ICGNN.predict(x_all, W_adj)

        intervention_mask = torch.zeros(self.processed_dim_all, device=self.device, dtype=torch.bool)
        log_p_x_base, log_p_u_base = self._log_prob_all(x_all, predict, intervention_mask)
        log_p_x_base = log_p_x_base.mean(0)
        log_p_u_base = log_p_u_base.mean(0)

        # Compute the remaining terms involved in the ELBO.
        log_p_A = self._log_prior_A(A_sample)
        log_q_A = -self.var_dist_A.entropy()
        penalty_dag = self.dagness_factor(A_sample)

        return {
            "penalty_dag": penalty_dag,
            "log_p_A": log_p_A,
            "log_p_x_base": log_p_x_base,
            "log_p_u_base": log_p_u_base,
            "log_q_A": log_q_A * factor_q,
            "log_qu_x": log_qu_x * factor_q,
        }

    def compute_loss(
        self,
        step: int,
        x: torch.Tensor,
        mask_train_batch: torch.Tensor,
        input_mask: torch.Tensor,
        num_samples: int,
        tracker: dict,
        train_config_dict: dict[str, Any],
        alpha: float = None,
        rho: float = None,
        adj_true: Optional[np.ndarray] = None,
        compute_cd_fscore: bool = False,
        bidirected_adj_true: Optional[np.ndarray] = None,
        beta: float = 1.0,
        **kwargs,
    ) -> tuple[torch.Tensor, dict]:
        """Computes loss and updates trackers of different terms.

        Args:
            step: Current iteration of the training loop.
            x: Batched samples from the dataset, size (batch_size, input_dim).
            mask_train_batch: Indicates which values are missing in the dataset, size (batch_size, input_dim).
            input_mask: Indicates which values are artificially masked, size (batch_size, input_dim).
            num_samples: Number of samples used to estimate the ELBO.
            tracker: Tracks the terms involved in the ELBO.
            train_config_dict: Contains the current training configuration.
            alpha: DAG enforcing prior parameter.
            rho: DAG enforcing prior parameter.
            adj_true: ground truth adj matrix.
            compute_cd_fscore: whether to compute `directed_fscore` and `bidirected_fscore` metrics at each step.
            bidirected_adj_true: ground truth bidirected adj matrix
            beta: KL term annealing coefficient.

        Returns:
            Loss and trackers of different terms.
        """
        _ = kwargs

        if self.imputation:
            # Get mean for imputation with artificially masked data, "regularizer" for imputation network
            _, _, mean_rec = self.impute_and_compute_entropy(x, input_mask)
            # Compute reconstruction loss on artificially dropped data
            scoring_mask = mask_train_batch
            reconstruction_term = scoring_mask * (mean_rec - x) ** 2  # Shape (batch_size, input_dim)
            reconstruction_term = reconstruction_term.sum(dim=1)  # Shape (batch_size)
            # Fill in missing data with amortized variational approximation for causality (without artificially masked data)
            x_fill, entropy_filled_term, _ = self.impute_and_compute_entropy(
                x, mask_train_batch
            )  # x_fill (batch_size, input_dim)  entropy  (batch_size)
            imputation_entropy = entropy_filled_term.mean(dim=0)  # (1)
            avg_reconstruction_err = reconstruction_term.mean(dim=0)
        else:
            x_fill = x
            imputation_entropy = avg_reconstruction_err = torch.tensor(0.0, device=self.device)

        #  Compute remaining terms
        elbo_terms = self._ELBO_terms(x_fill)
        log_p_x_base = elbo_terms["log_p_x_base"].mean(dim=0)
        log_p_u_base = elbo_terms["log_p_u_base"].mean(dim=0)
        log_p_A = elbo_terms["log_p_A"] / num_samples
        log_q_A = elbo_terms["log_q_A"] / num_samples
        log_qu_x = elbo_terms["log_qu_x"].mean(dim=0)

        log_p_u_weighed = log_p_u_base * beta
        log_qu_x_weighed = log_qu_x * beta

        penalty_dag = elbo_terms["penalty_dag"] * alpha / num_samples
        penalty_dag += elbo_terms["penalty_dag"] * elbo_terms["penalty_dag"] * rho / (2 * num_samples)

        if train_config_dict["anneal_entropy"] == "linear":
            elbo = (
                log_p_x_base
                + log_p_u_weighed
                + imputation_entropy
                + log_p_A
                - log_q_A / max(5 - step, 1)
                - log_qu_x_weighed / max(5 - step, 1)
                - penalty_dag
            )
        elif train_config_dict["anneal_entropy"] == "noanneal":
            elbo = (
                log_p_x_base + log_p_u_weighed + imputation_entropy + log_p_A - log_q_A - log_qu_x_weighed - penalty_dag
            )

        loss = -elbo + avg_reconstruction_err * train_config_dict["reconstruction_loss_factor"]

        if adj_true is not None and compute_cd_fscore:
            if bidirected_adj_true is None:
                bidirected_adj_true = np.zeros_like(adj_true["directed_adj_true"])

            directed_adj, bidirected_adj = cast(Any, self).get_admg_matrices(most_likely_graph=True, samples=1)
            directed_adj = directed_adj.astype(float).round()
            bidirected_adj = bidirected_adj.astype(float).round()
            bidirected_results = edge_prediction_metrics_multisample(
                bidirected_adj_true, bidirected_adj, adj_matrix_mask=None
            )

            directed_results = edge_prediction_metrics_multisample(adj_true, directed_adj, adj_matrix_mask=None)
            tracker["directed_fscore"].append(directed_results["adjacency_fscore"])
            tracker["bidirected_fscore"].append(bidirected_results["adjacency_fscore"])

        tracker["loss"].append(loss.item())
        tracker["penalty_dag"].append(elbo_terms["penalty_dag"].item())
        tracker["penalty_dag_weighed"].append(penalty_dag.item())
        tracker["log_p_A_sparse"].append(log_p_A.item())
        tracker["log_p_x"].append(log_p_x_base.item())
        tracker["log_p_u"].append(log_p_u_base.item())
        tracker["log_p_u_weighed"].append(log_p_u_weighed.item())
        tracker["imputation_entropy"].append(imputation_entropy.item())
        tracker["log_q_A"].append(log_q_A.item())
        tracker["log_qu_x"].append(log_qu_x.item())
        tracker["log_qu_x_weighed"].append(log_qu_x_weighed.item())
        tracker["reconstruction_mse"].append(avg_reconstruction_err.item())
        return loss, tracker

    def process_dataset(
        self,
        dataset: Dataset,
        train_config_dict: Optional[dict[str, Any]] = None,
        variables: Optional[Variables] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generates the training data and mask.

        Args:
            dataset: Dataset to use.
            train_config_dict: Dictionary with training hyperparameters.
            variables: Variables contained in the dataset.

        Returns:
            Output of DECI.process_dataset().
        """
        if variables is None:
            variables = self.observed_variables

        return super().process_dataset(dataset, train_config_dict, variables)


class ADMGParameterisedDDECI(DDECI):
    """An acyclic directed mixed graph (ADMG) parameterisation of the causal graph for D-DECI.

    The difference between this and DDECI is that this uses the ADMG parameterisation, which means that we don't need
    to specify the number of latent variables or enforce constraints on the latent variables being confounders.
    """

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        *args,
        **kwargs,
    ):
        """
        Args:
            observed_variables: Information about the observed variables / features used by this model.
            *args: Additional arguments to be passed to DECI.__init__().
            **kwargs: Additional keyword arguments to be passed to DECI.__init__().
        """
        # Construct latent variables for every pair of variable groups.
        d = variables.num_groups
        all_variables = _add_latent_variables(variables, d * (d - 1) // 2)
        super().__init__(model_id, all_variables, *args, **kwargs)

        if isinstance(self.var_dist_A, VarDistA_ENCO):
            # Overwrite self.var_dist_A.
            self.var_dist_A: VarDistA_ENCO_ADMG = VarDistA_ENCO_ADMG(
                device=self.device,
                input_dim=self.observed_num_nodes,
                tau_gumbel=self.tau_gumbel,
                dense_init=self.dense_init,
            )

    @classmethod
    def name(cls) -> str:
        return "admg_ddeci"

    def get_admg_matrices_tensor(
        self, do_round: bool = True, samples: int = 100, most_likely_graph: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Samples from the variational distribution over ADMGs.

        Args:
            do_round: Whether to round the edge probabilities. Defaults to True.
            samples: Number of samples to draw. Defaults to 100.
            most_likely_graph: Whether to sample the most likely graph. Defaults to False.

        Raises:
            NotImplementedError: If adjacency mode is not found.

        Returns:
            The directed and bidirected adjacency matrices.
        """
        if self.mode_adjacency == "learn":
            if most_likely_graph:
                assert samples == 1, "When passing most_likely_graph, only 1 sample can be returned."
                if hasattr(self.var_dist_A, "get_bidirected_adj_matrix"):
                    directed_adj_samples = [self.var_dist_A.get_directed_adj_matrix(do_round=do_round)]
                    bidirected_adj_samples = [self.var_dist_A.get_bidirected_adj_matrix(do_round=do_round)]
                else:
                    adj = self.var_dist_A.get_adj_matrix()
                    directed_adj, bidirected_adj = dag2admg(adj)
                    directed_adj_samples = [directed_adj]
                    bidirected_adj_samples = [bidirected_adj]
            else:
                if hasattr(self.var_dist_A, "sample_bidirected_adj"):
                    directed_adj_samples = [self.var_dist_A.sample_directed_adj() for _ in range(samples)]
                    bidirected_adj_samples = [self.var_dist_A.sample_bidirected_adj() for _ in range(samples)]
                else:
                    adj_samples = [self.var_dist_A.sample_A() for _ in range(samples)]
                    directed_adj_samples, bidirected_adj_samples = [], []
                    for adj in adj_samples:
                        directed_adj, bidirected_adj = dag2admg(adj)
                        directed_adj_samples.append(directed_adj)
                        bidirected_adj_samples.append(bidirected_adj)

                if do_round:
                    directed_adj_samples = [directed_adj.round() for directed_adj in directed_adj_samples]
                    bidirected_adj_samples = [bidirected_adj.round() for bidirected_adj in bidirected_adj_samples]

            directed_adj = torch.stack(directed_adj_samples, dim=0)
            bidirected_adj = torch.stack(bidirected_adj_samples, dim=0)
        else:
            raise NotImplementedError(f"Adjacency mode {self.mode_adjacency} not implemented")

        return directed_adj, bidirected_adj

    def get_admg_matrices(
        self,
        do_round: bool = True,
        samples: int = 100,
        most_likely_graph: bool = False,
        squeeze: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the directed and bidirected matrix (or several) as a numpy array.
        """
        directed_adj, bidirected_adj = self.get_admg_matrices_tensor(do_round, samples, most_likely_graph)

        if squeeze and samples == 1:
            directed_adj = directed_adj.squeeze(0)
            bidirected_adj = bidirected_adj.squeeze(0)
        # Here we have the cast to np.float64 because the original type
        # np.float32 has some issues with json, when saving causality results
        # to a file.
        return (
            directed_adj.detach().cpu().numpy().astype(np.float64),
            bidirected_adj.detach().cpu().numpy().astype(np.float64),
        )


class BowFreeDDECI(ADMGParameterisedDDECI):
    """An acyclic directed mixed graph (ADMG) parameterisation of the causal graph for D-DECI, with an additional
    bow-free constraint.

    The bow-free constraint provides structural identifiablity for D-DECI, and is enforced by replacing the DAG
    constraint of DECI with the bow-free constraint introduced in http://proceedings.mlr.press/v130/bhattacharya21a/bhattacharya21a.pdf.
    """

    @classmethod
    def name(cls) -> str:
        return "bowfree_ddeci"

    def dagness_factor(self, A: torch.Tensor) -> torch.Tensor:
        """Computes the dag + bow-free penalty for matrix A as trace(expm(D)) - dim + sum(D . B).

        Args:
            A: Magnified adjency matrix, size (self.processed_dim_all, self.processed_dim_all).

        Returns:
            The penalty term.
        """
        directed_adj, bidirected_adj = self.var_dist_A.demagnify_adj_matrix(A)
        return (
            torch.trace(torch.matrix_exp(directed_adj))
            - self.observed_num_nodes
            + torch.sum(directed_adj * bidirected_adj)
        )
