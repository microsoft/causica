from __future__ import annotations  # Needed to support method returning instance of its parent class until Python 3.10

import itertools
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, TypeVar, Union

import graphviz
import numpy as np
import pandas as pd
import torch
from dowhy import CausalModel
from dowhy.causal_identifier.auto_identifier import METHOD_NAMES
from joblib import Parallel, delayed
from scipy.stats import multivariate_normal
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PolynomialFeatures

from ..datasets.dataset import Dataset
from ..datasets.variables import Variables
from ..models.imodel import IModelForInterventions
from ..models.model import Model
from ..utils.io_utils import save_json, save_txt

T = TypeVar("T", bound="DoWhy")


class DoWhy(Model, IModelForInterventions):
    "TODO: deduplicate these file paths between DoWhy, CastleCausalLearner and SkLearnImputer"
    _model_config_path = "model_config.json"
    _model_type_path = "model_type.txt"
    _variables_path = "variables.json"
    model_file = "model.pt"

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        linear: bool = False,
        polynomial_order: int = 2,
        polynomial_bias: bool = True,
        bootstrap_samples: int = 100,
        adj_matrix: Optional[np.ndarray] = None,
        train_data: Optional[np.ndarray] = None,
        causal_identifier_method: str = "minimal-adjustment",
        adj_matrix_samples: Optional[List[np.ndarray]] = None,
        adj_matrix_sample_weights: Optional[np.ndarray] = None,
        random_seed: int = 0,
        parallel_n_jobs: int = 1,
    ):
        # TODO: implement methods for binary treatment settings
        """
        Wrapper class for Do-Why (https://microsoft.github.io/dowhy/). Takes care of graph loading,
             graph marginalisation, and exposes methods to compute CATEs and interventional distribution Log Likelihoods.

            If no graph is set at the moment of training, the true graph will be used.

        Args:
            model_id: Unique model ID for referencing this model instance.
            variables: Information about variables/features used by this model.
            save_dir: Location to save any information about this model, including training data. This wrapper is stateless but this information is needed for test result saving purposes.
            linear: whether to run backdoor.linear inference (True) or backdoor.DML (False)
            polynomial_order: When linear=False, this is the order of the featurisation of the conditional variables which goes into the final TE estimation step of DML,
            polynomial_bias:  When linear=False, this determines whether to include a bias term in the featurisation of the conditional variables which goes into the final TE estimation step of DML,
            bootstrap_samples: How many samples to use for uncertainty estimation when it is not computable in closed form, i.e. when linear=False
            adj_matrix: most likely adjacency matrix
            train_data: observational data from non-intervened system
            causal_identifier_method: Method name for DoWhy backdoor adjustment set. Should be one of {default, exhaustive-search, minimal-adjustment, maximal-adjustment}.
            adj_matrix_samples: list of samples from adjacency matrix posterior
            adj_matrix_sample_weights: weight for samples of adjacency matrix posterior. If unspecified, uniform weighing (1/Nsamples) will be used
            random_seed: used determine convergence of non-linear models that rely on non-convex optimisation and can give different results across seeds
            parallel_n_jobs: number of parallel jobs to use when running DoWhy. The default 1 disables parallelism. -1 uses as many threads as there are CPUs.
        """
        super().__init__(model_id, variables, save_dir)

        self.labels = self.variables.group_names
        self.categorical_columns = [
            group
            for group, idxs in zip(self.variables.group_names, self.variables.group_idxs)
            if self.variables[idxs[0]].type_ in ["categorical", "binary"]
        ]

        # Load Causal DAG directly
        self.adj_matrix = adj_matrix
        self.graph: Union[str, None]
        if adj_matrix is not None:
            assert adj_matrix.shape[0] == len(variables)
            self.graph = DoWhy.str_to_dot(self.graph_from_adjacency(adj_matrix).source)
        else:
            self.graph = None

        self.adj_matrix_samples = adj_matrix_samples
        self.graph_samples: Union[list, None]
        if adj_matrix_samples is not None:
            self.graph_samples = [DoWhy.str_to_dot(self.graph_from_adjacency(mat).source) for mat in adj_matrix_samples]
            if adj_matrix_sample_weights is None:
                adj_matrix_sample_weights = np.ones(len(adj_matrix_samples)) / len(adj_matrix_samples)

            assert np.abs(adj_matrix_sample_weights.sum() - 1) < 1e-3
        else:
            self.graph_samples = None

        self.adj_matrix_sample_weights = adj_matrix_sample_weights

        # DoWhy parameters
        self._linear = linear
        self._polynomial_order = polynomial_order
        self._polynomial_bias = polynomial_bias
        self._bootstrap_samples = bootstrap_samples
        self._train_data = train_data
        self._random_seed = random_seed
        self.parallel_n_jobs = parallel_n_jobs

        method_names = {method_name.value for method_name in METHOD_NAMES}
        assert (
            causal_identifier_method in method_names
        ), f"causal_identifier_method {causal_identifier_method} is unknown."
        self._causal_identifier_method = causal_identifier_method

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Optional[Dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """This method just loads up the train data. If no graph has been loaded, it also loads the ground truth graph"""
        self._train_data, _ = dataset.train_data_and_mask
        assert self._train_data is not None
        self._train_data = self._train_data.astype(np.float32)

        assert self.graph is not None, "the graph should have been loaded before calling the train method"

    def _normalise_inputs(
        self,
        intervention_idxs: np.ndarray,
        intervention_values: np.ndarray,
        conditioning_idxs: Optional[np.ndarray] = None,
        conditioning_values: Optional[np.ndarray] = None,
        reference_values: Optional[np.ndarray] = None,
    ):
        """
        Normalise train data, intervention data and conditioning data
        """
        # Normalise values between 0 and 1.
        # TODO 18375: can we avoid the repeated (un)normalization of data before/during this function or at least
        # share the normalization logic in both places?
        lowers = np.array([var.lower for var in self.variables if var.type_ != "text"])
        uppers = np.array([var.upper for var in self.variables if var.type_ != "text"])
        assert isinstance(self._train_data, np.ndarray)
        normed_data = np.subtract(self._train_data.copy(), lowers) / (uppers - lowers)
        train_df = pd.DataFrame(normed_data, columns=self.labels)
        intervention_dtype = intervention_values.dtype
        intervention_values = np.divide(intervention_values.copy() - lowers[intervention_idxs], uppers - lowers)[
            intervention_idxs
        ]
        # TODO: Remove forced dtype after normalization due to incompatibility with non-int intervention values.
        intervention_values = intervention_values.astype(dtype=intervention_dtype)
        if reference_values is not None:
            reference_values = np.divide(reference_values.copy() - lowers[intervention_idxs], uppers - lowers)[
                intervention_idxs
            ]
        if conditioning_values is not None:
            conditioning_values = np.divide(conditioning_values.copy() - lowers[conditioning_idxs], uppers - lowers)[
                conditioning_idxs
            ]

        return (
            train_df,
            intervention_idxs,
            intervention_values,
            conditioning_idxs,
            conditioning_values,
            reference_values,
        )

    def _dowhy_cate(
        self,
        graph: str,
        train_df: pd.DataFrame,
        intervention_idxs: np.ndarray,
        intervention_values: np.ndarray,
        reference_values: np.ndarray,
        target_idx: np.ndarray,
        conditioning_idxs: Optional[np.ndarray] = None,
        conditioning_values: Optional[np.ndarray] = None,
        get_std_err: bool = True,
        fixed_seed: Optional[int] = None,
    ):
        """
            Compute (optionally Conditional) Average Treatment Effects for a single treatment variable .
            That is E[Y | do(T=k), X] - E[Y | X] (or, optionally, E[Y | do(T=k), X] - E[Y | do(T=k'), X]). Y is the target variable
            T is the treatment variable, and X is an optional conditioning variable.
            This is the workhorse method of our DoWhy wrapper.

        Args:
            graph: string containing adjacency matrix in DOT format
            train_df: pd.DataFrame containing the values of each variable
            intervention_idxs: np.ndarray containing indices of variables that have been intervened.
            intervention_values: np.ndarray containing values for variables that have been intervened.
            target_idx: np.ndarray containing the index of the target variable.
            reference_values: np.ndarray containing reference value for the treatment. Will compute
                E[Y | do(T=k), X] - E[Y | do(T=k'), X].
            conditioning_idxs: optional np.ndarray containing indices of variables that we condition on.
            conditioning_values: optional np.ndarray array containing values for variables that we condition on.
            get_std_err: whether to compute errorbars. If they are not needed, setting this to false can avoid costly boostrap computations.
            normalise: whether to perform maxmin normalisation
            fixed_seed: whether to run with a specific numpy seed. This ensures the same local optima is reached for different outcome dimensions and graph samples.
        Returns:
            (np.ndarray, np.ndarray): containing the conditional mean of the treated variables and their standard error
        """

        if fixed_seed is not None:
            np.random.seed(fixed_seed)

        if intervention_idxs[0] == target_idx:
            return intervention_values[0] - reference_values[0], np.nan

        if conditioning_idxs is not None and conditioning_values is not None:
            assert len(conditioning_idxs) == 1 and len(conditioning_values) == 1, "Only single conditions are supported"
            conditioning_dict = {self.labels[conditioning_idxs[0]]: conditioning_values[0]}
        else:
            conditioning_dict = None

        dowhy_model = CausalModel(
            data=train_df,
            treatment=self.labels[intervention_idxs[0]],
            outcome=self.labels[target_idx],
            graph=graph,
        )

        identified_estimand = dowhy_model.identify_effect(
            proceed_when_unidentifiable=True, method_name=self._causal_identifier_method
        )

        outcome_is_category = train_df.dtypes[target_idx].name == "category"
        if outcome_is_category:
            raise NotImplementedError("Our DoWhy wrapper currently requires a continuous target variable.")

        if not self._linear:
            treatment_is_category = train_df.dtypes[intervention_idxs[0]].name == "category"
            estimate = self._econml_dml_estimate_effect(
                dowhy_model,
                identified_estimand,
                reference_values,
                intervention_values,
                conditioning_dict,
                treatment_is_category,
            )
        else:
            estimate = dowhy_model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                control_value=reference_values[0],
                treatment_value=intervention_values[0],
                confidence_intervals=(
                    get_std_err and conditioning_dict is None
                ),  # only get errorbars if they are requested and we are not computing CATE. (This is due to bug in CATE errorbars)
                effect_modifiers=conditioning_dict,
            )

        target_col = train_df[self.labels[target_idx]]
        if estimate.value is None:  # Case where there is no relation in the graph
            mean = 0.0
            std_err = target_col.std() / (train_df.shape[0] ** 0.5)
        else:
            mean = estimate.value
            if get_std_err:
                if not hasattr(
                    estimate, "estimator"
                ):  # if the errorbars were not previously computed return the variance in the train data.
                    std_err = target_col.std() / (train_df.shape[0] ** 0.5)
                else:
                    std_err = (
                        estimate.get_standard_error()
                    )  # this corresponds to stdev directly (https://www.statsmodels.org/dev/_modules/statsmodels/regression/linear_model.html#RegressionResults)
                    if not np.isscalar(std_err):  # workaround for strange inconsistent behaviour
                        std_err = std_err[0]
            else:
                std_err = np.nan

        return mean, std_err

    def _econml_dml_estimate_effect(
        self,
        dowhy_model,
        identified_estimand,
        reference_values,
        intervention_values,
        conditioning_dict,
        treatment_is_category=False,
    ):
        """
        Estimate CATE using EconML's DML method. The choice of models is determined by the treatment (intervention) and outcome (target)
        variable types.
        """

        if treatment_is_category:
            model_t = GradientBoostingClassifier()
        else:
            model_t = GradientBoostingRegressor()

        return dowhy_model.estimate_effect(
            identified_estimand,
            method_name="backdoor.econml.dml.DML",
            control_value=reference_values.item(0),
            treatment_value=intervention_values.item(0),
            confidence_intervals=False,  # False because the computation is too costly for us
            effect_modifiers=conditioning_dict,
            method_params={
                "init_params": {
                    "model_y": GradientBoostingRegressor(),
                    "model_t": model_t,
                    "model_final": LassoCV(fit_intercept=False),
                    "featurizer": PolynomialFeatures(
                        degree=self._polynomial_order, include_bias=self._polynomial_bias
                    ),  # this featurizes the variables we condition on
                    "discrete_treatment": treatment_is_category,
                },
                "fit_params": {  # we dont use bootstrap for now, as there seems to be a bug in how std error is calculated
                    # 'inference': BootstrapInference(n_bootstrap_samples=100, n_jobs=-1)
                },
            },
        )

    def _dowhy_inference(
        self,
        graphs: List[str],
        intervention_idxs: np.ndarray,
        intervention_values: np.ndarray,
        reference_values: Optional[np.ndarray] = None,
        effect_idxs: Optional[np.ndarray] = None,
        conditioning_idxs: Optional[np.ndarray] = None,
        conditioning_values: Optional[np.ndarray] = None,
        get_std_err: bool = True,
        normalise: bool = False,
        fixed_seed: Optional[int] = None,
    ):
        """
            Compute (optionally Conditional) Average Treatment Effects for every non-intervened (non-treated) variable under consideration.
            That is E[Y | do(T=k), X] - E[Y | X] (or, optionally, E[Y | do(T=k), X] - E[Y | do(T=k'), X]). Y is the outcome of interest
            T is the treatment variable, and X is an optional conditioning variable.
            This method compute the (C)ATE for treating each variable in turn as a target variable.
            This method also parallelises over multiple weighted graphs, giving the (C)ATE from the different possible graphs.

        Args:
            graphs: a list of strings, each string containinga graph adjacency matrix in DOT format.
            intervention_idxs: np.ndarray containing indices of variables that have been intervened.
            intervention_values: np.ndarray containing values for variables that have been intervened.
            conditioning_idxs: optional np.ndarray containing indices of variables that we condition on.
            conditioning_values: optional np.ndarray array containing values for variables that we condition on.
            reference_values: optional np.ndarray containing a reference value for the treatment. If specified will compute
                E[Y | do(T=k), X] - E[Y | do(T=k'), X]. Otherwise will compute E[Y | do(T=k), X] - E[Y | X] = E[Y | do(T=k), X] - E[Y | do(T=mu_T), X]
            effect_idxs:  optional np.ndarray containing indices on which treatment effects should be evaluated.
                If None, every variable that is not used for treatment or conditioning will be used for evaluation.
            get_std_err: whether to compute errorbars. If they are not needed, setting this to false can avoid costly boostrap computations.
            normalise: whether to perform maxmin normalisation
            fixed_seed: whether to run with a specific numpy seed. This ensures the same local optima is reached for different outcome dimensions and graph samples.
        Returns:
            (np.ndarray, np.ndarray): containing the conditional mean of the treated variables and their standard error
        """

        assert len(intervention_idxs) == 1, "DoWhy only supports single variable interventions"
        assert len(intervention_values) == 1, "DoWhy only supports single variable interventions"
        assert (
            reference_values is None or len(reference_values) == 1
        ), "DoWhy only supports single variable interventions, multiple reference values passed"

        if fixed_seed is not None:
            np.random.seed(fixed_seed)
        else:
            np.random.seed(self._random_seed)

        if normalise:
            (
                train_df,
                intervention_idxs,
                intervention_values,
                conditioning_idxs,
                conditioning_values,
                reference_values,
            ) = self._normalise_inputs(
                intervention_idxs, intervention_values, conditioning_idxs, conditioning_values, reference_values
            )
        else:
            train_df = pd.DataFrame(self._train_data, columns=self.labels)

        # Set the correct pandas type for categorical variables
        train_df = train_df.astype({name: "category" for name in self.categorical_columns}, copy=False)

        if reference_values is None:
            # TODO: when implmenting CATE, will need to compute conditional mean instead
            if self.labels[intervention_idxs[0]] not in self.categorical_columns:
                reference_values = np.array([train_df[self.labels[intervention_idxs[0]]].mean()])
            else:
                # Pick an arbitrary reference
                reference_values = np.array([train_df[self.labels[intervention_idxs[0]]][0]])

        # parallel dowhy estimation over outcome (target) dimensions and graphs
        target_optns: Union[np.ndarray, Iterable[int]]
        if effect_idxs is not None:
            target_optns = effect_idxs
        else:
            target_optns = range(len(self.variables))

        product_space = itertools.product(target_optns, graphs)

        seed_multiplier = len(effect_idxs) if effect_idxs is not None else len(self.variables)
        inner_seeds = np.random.randint(2**20, size=seed_multiplier * len(graphs))

        outputs = Parallel(n_jobs=self.parallel_n_jobs, backend="multiprocessing")(
            delayed(self._dowhy_cate)(
                graph,
                train_df,
                intervention_idxs=intervention_idxs,
                intervention_values=intervention_values,
                reference_values=reference_values,
                target_idx=target,
                conditioning_idxs=conditioning_idxs,
                conditioning_values=conditioning_values,
                get_std_err=get_std_err,
                fixed_seed=inner_seed,
            )
            for ((target, graph), inner_seed) in zip(product_space, inner_seeds)
        )

        # Split out the results
        means_list, stds_list = zip(*outputs)
        means, stds = np.array(means_list), np.array(stds_list)

        # Aggregate the results by weighted sum over graphs
        # Dimension 0 = targets, Dimension 1 = graphs
        means, stds = means.reshape(len(target_optns), -1), stds.reshape(len(target_optns), -1)

        return means, stds

    def cate(
        self,
        intervention_idxs: Union[torch.Tensor, np.ndarray],
        intervention_values: Union[torch.Tensor, np.ndarray],
        reference_values: Optional[np.ndarray] = None,
        effect_idxs: Optional[np.ndarray] = None,
        conditioning_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        conditioning_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        Nsamples_per_graph: int = 1,
        Ngraphs: Optional[int] = 1000,
        most_likely_graph: bool = False,
        fixed_seed: Optional[int] = None,
    ):
        """
        Returns average treatment effect for a given intervention. Optionally, can condition on additional variables

        Args:
            intervention_idxs: np.ndarray containing indices of variables that have been intervened.
            intervention_values: np.ndarray containing values for variables that have been intervened.
            conditioning_idxs: optional np.ndarray containing indices of variables that we condition on.
            conditioning_values: optional np.ndarray array containing values for variables that we condition on.
            reference_values: optional np.ndarray containing a reference value for the treatment. If specified will compute
                E[Y | do(T=k), X] - E[Y | do(T=k'), X]. Otherwise will compute E[Y | do(T=k), X] - E[Y | X] = E[Y | do(T=k), X] - E[Y | do(T=mu_T|X), X]
            effect_idxs:  optional np.ndarray containing indices on which treatment effects should be evaluated
            Nsamples_per_graph: dummy input needed to match signature used in evaluation code
            Ngraphs:  dummy input needed to match signature used in evaluation code
            most_likely_graph: bool indicatng whether to deterministically pick the most probable graph under the approximate posterior or to draw a new graph for every sample

        Returns:
            (ate, ate_norm): (np.ndarray, np.ndarray) both of size (input_dim) average treatment effect computed on regular and normalised data
        """
        assert isinstance(intervention_idxs, np.ndarray)
        assert isinstance(intervention_values, np.ndarray)
        if conditioning_idxs is not None:
            assert isinstance(conditioning_idxs, np.ndarray)
        if conditioning_values is not None:
            assert isinstance(conditioning_values, np.ndarray)

        if most_likely_graph:
            assert isinstance(self.graph, str)
            graphs = [self.graph]
            weights = np.array([1.0])
        else:
            if self.adj_matrix_samples is not None:
                assert self.graph_samples is not None
                graphs = self.graph_samples
            else:
                assert isinstance(self.graph, str)
                graphs = [self.graph]

            weights = self.adj_matrix_sample_weights if self.adj_matrix_samples is not None else np.array([1.0])  # type: ignore

        cate, _ = self._dowhy_inference(
            graphs,
            intervention_idxs=intervention_idxs,
            intervention_values=intervention_values,
            reference_values=reference_values,
            effect_idxs=effect_idxs,
            conditioning_idxs=conditioning_idxs,
            conditioning_values=conditioning_values,
            get_std_err=False,
            normalise=False,
            fixed_seed=fixed_seed,
        )

        norm_cate, _ = self._dowhy_inference(
            graphs,
            intervention_idxs=intervention_idxs,
            intervention_values=intervention_values,
            reference_values=reference_values,
            effect_idxs=effect_idxs,
            conditioning_idxs=conditioning_idxs,
            conditioning_values=conditioning_values,
            get_std_err=False,
            normalise=True,
            fixed_seed=fixed_seed,
        )

        weights = weights.reshape(1, -1)
        cate_aggregated = (cate * weights).sum(1)
        norm_cate_aggregated = (norm_cate * weights).sum(1)

        return cate_aggregated, norm_cate_aggregated

    def sample(
        self,
        Nsamples: int = 100,
        most_likely_graph: bool = False,
        intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        intervention_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        raise NotImplementedError()

    def log_prob(
        self,
        X: Union[torch.Tensor, np.ndarray],
        intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        intervention_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        conditioning_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        conditioning_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        Nsamples_per_graph: int = 1,
        Ngraphs: Optional[int] = 1000,
        most_likely_graph: bool = False,
        fixed_seed: Optional[int] = None,
    ):

        """
        Evaluate log probability of test samples under an intervention distribution. DoWhy does not provide intervention distributions. However,
        In the presence of multimodality induced by uncertainty over graphs, a distributional approach is more reliable than an expectation.
        The mean of this intervention distribution is computed as E[Y | do(T=k), X] - E[Y | do(T=mu_T|X), X] + E[Y | X], where the first two terms are provided
        by DoWhy and the last term is estimated directly from the train data. The standard deviation is obtained from the standard error of the mean estimator.
        We make a parametric normal assumption about p(Y | do(T), X). In the case of multiple plausible graphs, we use a GMM.

        Args:
            X: np.ndarray of observations for which to compute log-probability
            intervention_idxs: np.ndarray containing indices of variables that have been intervened.
            intervention_values: np.ndarray containing values for variables that have been intervened.
            conditioning_idxs: optional np.ndarray containing indices of variables that we condition on.
            conditioning_values: optional np.ndarray array containing values for variables that we condition on.
            Nsamples: dummy input needed to match signature used in evaluation code
            most_likely_graph: bool indicatng whether to deterministically pick the most probable graph under the approximate posterior or to draw a new graph for every sample

        Returns:
            log_prob: torch.tensor  (Nsamples)
        """
        assert isinstance(intervention_idxs, np.ndarray)
        assert isinstance(intervention_values, np.ndarray)
        if conditioning_idxs is not None:
            assert isinstance(conditioning_idxs, np.ndarray)
        if conditioning_values is not None:
            assert isinstance(conditioning_values, np.ndarray)

        assert self._train_data is not None
        assert isinstance(self.graph, str)
        Ntrain = self._train_data.shape[0]
        # this will have to be a conditional mean if we have conditioning variables
        target_mean = self._train_data.mean(axis=0)
        if most_likely_graph:
            graphs = [self.graph]
        else:
            graphs = self.graph_samples if self.adj_matrix_samples is not None else [self.graph]  # type: ignore
            weights = self.adj_matrix_sample_weights if self.adj_matrix_samples is not None else np.array([1.0])

        cate_samples, std_err_samples = self._dowhy_inference(
            graphs,
            intervention_idxs=intervention_idxs,
            intervention_values=intervention_values,
            reference_values=None,
            effect_idxs=None,
            conditioning_idxs=conditioning_idxs,
            conditioning_values=conditioning_values,
            get_std_err=True,
            normalise=False,
            fixed_seed=fixed_seed,
        )

        if most_likely_graph:

            cate, std_err = cate_samples.squeeze(1), std_err_samples.squeeze(1)

            mean = cate + target_mean
            var = (std_err**2) * Ntrain

            assert mean.ndim == 1
            assert var.ndim == 1

            # TODO: add conditional values here
            X = np.delete(X, intervention_idxs, axis=1)
            mean = np.delete(mean, intervention_idxs, axis=0)
            var = np.delete(var, intervention_idxs, axis=0)

            log_probs = multivariate_normal.logpdf(X, mean, var)

        else:

            cate_samples, std_err_samples = cate_samples.T, std_err_samples.T
            mean_samples = cate_samples + target_mean[None, :]
            var_samples = (std_err_samples**2) * Ntrain
            Ncomponents = cate_samples.shape[0] - 1

            X = np.delete(X, intervention_idxs, axis=1)
            mean_samples = np.delete(mean_samples, intervention_idxs, axis=1)
            var_samples = np.delete(var_samples, intervention_idxs, axis=1)

            gmm = GaussianMixture(n_components=Ncomponents, covariance_type="diag")

            gmm.weights_ = weights  # self.adj_matrix_sample_weights  # mixture weights (n_components,)
            gmm.means_ = mean_samples  # mixture means (n_components, input_dim)
            gmm.covariances_ = var_samples  # mixture cov (n_components, input_dim)
            gmm.precisions_cholesky_ = var_samples ** (-0.5)
            gmm.converged_ = True

            log_probs = gmm.score_samples(X)

        return log_probs

    def load_graph_from_discovery_model(self, discovery_model, samples=5):
        """
        Load graphs from an instance of a pre-trained causal discovery model
        """
        self.adj_matrix = discovery_model.get_adj_matrix(samples=1, most_likely_graph=True, squeeze=True)
        assert self.adj_matrix.shape[0] == self.variables.num_groups
        self.graph = DoWhy.str_to_dot(self.graph_from_adjacency(self.adj_matrix).source)

        self.adj_matrix_samples = discovery_model.get_adj_matrix(samples=samples)

        self.adj_matrix_samples = [self.adj_matrix_samples[i, :, :] for i in range(self.adj_matrix_samples.shape[0])]
        self.graph_samples = [
            DoWhy.str_to_dot(self.graph_from_adjacency(mat).source) for mat in self.adj_matrix_samples
        ]
        if self.adj_matrix_sample_weights is None:
            self.adj_matrix_sample_weights = np.ones(len(self.adj_matrix_samples)) / len(self.adj_matrix_samples)

    def graph_from_adjacency(self, adjacency_matrix: np.ndarray):
        """
        Generate graphviz matrix string from numpy.ndarray adjacency matrix
        """
        idx = np.abs(adjacency_matrix) > 0.01
        dirs = np.where(idx)
        d = graphviz.Digraph(engine="dot")
        for name in self.labels:
            d.node(name)
        for to, from_, coef in zip(dirs[1], dirs[0], adjacency_matrix[idx]):
            d.edge(self.labels[from_], self.labels[to], label=str(coef))
        return d

    @staticmethod
    def str_to_dot(string: str) -> str:
        """
        Converts input string from graphviz library to valid DOT graph format.
        """
        graph = string.replace("\n", ";").replace("\t", "")
        graph = graph[:9] + graph[10:-2] + graph[-1]  # Removing unnecessary characters from string
        return graph

    @classmethod
    def create(
        cls: Type[T],
        model_id: str,
        save_dir: str,
        variables: Variables,
        model_config_dict: Dict[str, Any],
        device: Union[str, int],
    ) -> T:

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model config to save dir.
        model_config_save_path = os.path.join(save_dir, cls._model_config_path)
        save_json(model_config_dict, model_config_save_path)

        # Save variables file.
        variables_path = os.path.join(save_dir, cls._variables_path)
        variables.save(variables_path)

        # Save model type.
        model_type_path = os.path.join(save_dir, cls._model_type_path)
        save_txt(cls.name(), model_type_path)

        return cls(model_id, variables, save_dir, **model_config_dict)

    @classmethod
    def name(cls) -> str:
        return "dowhy"

    @classmethod
    def load(cls: Type[T], model_id: str, save_dir: str, device: Union[str, int], **model_config_dict) -> T:

        variables_path = os.path.join(save_dir, cls._variables_path)
        variables = Variables.create_from_json(variables_path)

        return cls(model_id=model_id, variables=variables, save_dir=save_dir, **model_config_dict)

    def save(self) -> None:
        raise NotImplementedError()
