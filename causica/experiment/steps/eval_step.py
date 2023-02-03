"""
Run evaluation on a trained PVAE.

To run: python run_eval.py boston -ic parameters/impute_config.json -md runs/run_name/models/model_id
"""

import datetime as dt
import os
import warnings
from functools import partial
from logging import Logger
from typing import Any, Dict, Optional, Union, cast

import mlflow
import numpy as np
from scipy.sparse import csr_matrix, issparse

from ...datasets.dataset import (
    CausalDataset,
    Dataset,
    InterventionData,
    LatentConfoundedCausalDataset,
    SparseDataset,
    TemporalDataset,
)
from ...models.deci.ddeci import DDECI
from ...models.deci.deci import DECI
from ...models.imodel import IModelForCausalInference, IModelForCounterfactuals, IModelForImputation
from ...utils.causality_utils import (
    eval_temporal_causal_discovery,
    get_ate_rms,
    get_ite_evaluation_results,
    get_treatment_data_logprob,
)
from ...utils.evaluation_dataclasses import AteRMSEMetrics, IteRMSEMetrics, TreatmentDataLogProb
from ...utils.imputation import (
    eval_imputation,
    impute_targets_only,
    plot_pairwise_comparison,
    run_imputation_with_stats,
)
from ...utils.metrics import compute_target_metrics, save_confusion_plot, save_train_val_test_metrics
from ...utils.nri_utils import edge_prediction_metrics, edge_prediction_metrics_multisample
from ...utils.plot_functions import violin_plot_imputations
from ...utils.torch_utils import set_random_seeds

ALL_INTRVS = "all interventions"


def run_eval_main(
    model: IModelForImputation,
    dataset: Union[Dataset, SparseDataset],
    vamp_prior_data,
    impute_config: Dict[str, Any],
    extra_eval: bool,
    seed: int,
    split_type: str = "rows",
    user_id: int = 0,
    impute_train_data: bool = True,
):
    """
    Args:
        logger (`logging.Logger`): Instance of logger class to use.
        model (IModel): Model to use.
        dataset: Dataset or SparseDataset object.
        vamp_prior_data (tuple of numpy arrays): Tuple of (data, mask). Used for vamp prior samples.
        impute_config (dictionary): Dictionary containing options for inference.
        objective_config (dictionary): Dictionary containing objective configuration parameters.
        extra_eval (bool): If True, run evaluation that takes longer (pairwise comparison and information gain check.)
        split_type (str): Whether test data is split by rows ('rows') or elements ('elements'). If 'elements', the test
            set values will be predicted conditioned on the training set values.
        seed (int): Random seed to use when running eval.
        user_id (int): the index of the datapoint to plot the violin plot (uncertainty output).
        impute_train_data (bool): Whether imputation should be run on training data. This can require much more memory.
    """
    start_time = dt.datetime.utcnow()

    train_data, train_mask = dataset.train_data_and_mask
    val_data, val_mask = dataset.val_data_and_mask
    test_data, test_mask = dataset.test_data_and_mask

    # Assert that the test data has at least one row
    assert test_data is not None and test_mask is not None and dataset.variables is not None
    variables = dataset.variables
    assert test_data.shape == test_mask.shape
    user_count, _ = test_data.shape
    assert user_count > 0, "Empty test data array provided for evaluation"
    assert user_id < user_count, "Violin plot data point index out of bounds"
    assert split_type in ["rows", "elements"]

    # Fix evaluation seed
    set_random_seeds(seed)

    # Missing value imputation
    (
        _,
        _,
        _,
        train_metrics,
        _,
        _,
        _,
        val_metrics,
        test_obs_mask,
        test_target_mask,
        _,
        test_metrics,
    ) = eval_imputation(dataset, model, variables, split_type, vamp_prior_data, impute_config, impute_train_data, seed)

    save_train_val_test_metrics(
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        save_file=os.path.join(model.save_dir, "results.json"),
    )
    if len(variables.continuous_idxs) == 0:
        save_confusion_plot(test_metrics, model.save_dir)

    # Target value imputation
    if split_type != "elements":
        # Assume no target features if data split is elementwise
        if impute_train_data:
            train_target_imputations = impute_targets_only(
                model, train_data, train_mask, impute_config, vamp_prior_data
            )
            train_target_metrics = compute_target_metrics(train_target_imputations, train_data, variables)
        else:
            train_target_metrics = {}

        if val_data is None:
            val_target_metrics = {}
        else:
            val_target_imputations = impute_targets_only(model, val_data, val_mask, impute_config, vamp_prior_data)
            val_target_metrics = compute_target_metrics(val_target_imputations, val_data, variables)

        test_target_imputations = impute_targets_only(model, test_data, test_mask, impute_config, vamp_prior_data)
        test_target_metrics = compute_target_metrics(test_target_imputations, test_data, variables)

        save_train_val_test_metrics(
            train_metrics=train_target_metrics,
            val_metrics=val_target_metrics,
            test_metrics=test_target_metrics,
            save_file=os.path.join(model.save_dir, "target_results.json"),
        )
    else:
        train_target_metrics = {}
        val_target_metrics = {}
        test_target_metrics = {}

    # Log metrics to AzureML
    mlflow.log_metrics(train_metrics.get("all", {}))
    mlflow.log_metrics(val_metrics.get("all", {}))
    mlflow.log_metrics(test_metrics["all"])

    mlflow.log_metrics(train_metrics.get("Imputation MLL", {}))
    mlflow.log_metrics(val_metrics.get("Imputation MLL", {}))
    mlflow.log_metrics(test_metrics.get("Imputation MLL", {}))

    for val in train_target_metrics.values():
        mlflow.log_metrics(val)
    for val in val_target_metrics.values():
        mlflow.log_metrics(val)
    for val in test_target_metrics.values():
        mlflow.log_metrics(val)

    # Violin plot
    # Plot the violin plot output for one datapoint to check imputation results
    # Only plot for dense data, assume too many features if sparse
    if isinstance(test_obs_mask, np.ndarray):
        imputed_values, imputed_stats = run_imputation_with_stats(
            model, test_data, test_obs_mask, variables, impute_config, vamp_prior_data
        )
        if impute_config["sample_count"] < 10:
            warnings.warn("Imputation sample count is < 10, violin plot may be of low quality.")
        violin_plot_imputations(
            imputed_values,
            imputed_stats,
            test_target_mask,
            variables,
            user_id,
            title="Violin plot for continuous variables",
            save_path=model.save_dir,
            plot_name=f"imputed_values_violin_plot_user{user_id}",
            normalized=True,
        )

    # Extra evaluation
    if extra_eval:
        # Create pair plots
        # We assume all test_data is observed (i.e. no special treatment of input test mask)

        # For ground truth data
        if issparse(test_data):
            # Declare types to fix mypy error
            test_data_: csr_matrix = test_data
            test_data_dense = test_data_.toarray()
        else:
            test_data_dense = test_data
        plot_pairwise_comparison(
            test_data_dense, variables, filename_suffix="ground_truth_data", save_dir=model.save_dir
        )

        # Data generation
        # The impute() call is identical to one in AL beside averaging MC samples (which doesn't happen in AL)
        empty_mask = np.zeros_like(test_data_dense, dtype=bool)
        generated_values = model.impute(test_data_dense, empty_mask, impute_config, vamp_prior_data=vamp_prior_data)
        plot_pairwise_comparison(
            generated_values, model.variables, filename_suffix="data_generation", save_dir=model.save_dir
        )

        # Data reconstruction
        reconstructed_values = model.impute(test_data, test_mask, impute_config, vamp_prior_data=vamp_prior_data)
        plot_pairwise_comparison(
            reconstructed_values, model.variables, filename_suffix="data_reconstruction", save_dir=model.save_dir
        )

    mlflow.log_metric("impute/running-time", (dt.datetime.utcnow() - start_time).total_seconds() / 60)


def eval_causal_discovery(dataset: CausalDataset, model: IModelForCausalInference, disable_diagonal_eval: bool = True):
    """
    Args:
        logger (`logging.Logger`): Instance of logger class to use.
        dataset: Dataset or SparseDataset object.
        model (IModelForCausalInference): Model to use.
        conversion_type (str): It is used for temporal causal model evaluation. It supports "full_time" and "auto_regressive".
        "full_time" converts temporal adj matrix to full-time static graph and "auto_regressive" converts it to static graph only keeping
        the connections to the current timestep. For details, refer to docstring of `convert_temporal_adj_matrix_to_static`.

    This requires the model to have a method get_adjacency_data_matrix() implemented, which returns the adjacency
    matrix learnt from the data.
    """
    adj_ground_truth = dataset.get_adjacency_data_matrix()

    # For DECI, the default is to give 100 samples of the graph posterior
    adj_pred = model.get_adj_matrix().astype(float).round()

    # Convert temporal adjacency matrices to static adjacency matrices, currently does not support partially observed ground truth (i.e. subgraph_idx=None).
    if isinstance(dataset, TemporalDataset):
        results = eval_temporal_causal_discovery(dataset, model, disable_diagonal_eval=disable_diagonal_eval)
        if adj_ground_truth.ndim == 3:
            # Log metrics
            for suffix in ["overall_full_time", "overall_temporal", "inst", "lag"]:
                mlflow.log_metric(f"adjacency_{suffix}.recall", results.get(f"adjacency_recall_{suffix}", -1))
                mlflow.log_metric(f"adjacency_{suffix}.precision", results.get(f"adjacency_precision_{suffix}", -1))
                mlflow.log_metric(f"adjacency_{suffix}.f1", results.get(f"adjacency_fscore_{suffix}", -1), True)
                mlflow.log_metric(f"orientation_{suffix}.recall", results.get(f"orientation_recall_{suffix}", -1))
                mlflow.log_metric(f"orientation_{suffix}.precision", results.get(f"orientation_precision_{suffix}", -1))
                mlflow.log_metric(f"orientation_{suffix}.f1", results.get(f"orientation_fscore_{suffix}", -1), True)
                mlflow.log_metric(f"causal_accuracy_{suffix}", results.get(f"causal_accuracy_{suffix}", -1))
                mlflow.log_metric(f"causalshd_{suffix}", results.get(f"shd_{suffix}", -1))
                mlflow.log_metric(f"causalnnz_{suffix}", results.get(f"nnz_{suffix}", -1))
        elif adj_ground_truth.ndim == 2:
            mlflow.log_metric("adjacency_agg.recall", results.get("adjacency_recall_agg", -1))
            mlflow.log_metric("adjacency_agg.precision", results.get("adjacency_precision_agg", -1))
            mlflow.log_metric("adjacency_agg.f1", results.get("adjacency_fscore_agg", -1), True)
            mlflow.log_metric("orientation_agg.recall", results.get("orientation_recall_agg", -1))
            mlflow.log_metric("orientation_agg.precision", results.get("orientation_precision_agg", -1))
            mlflow.log_metric("orientation_agg.f1", results.get("orientation_fscore_agg", -1), True)
            mlflow.log_metric("causal_accuracy_agg", results.get("causal_accuracy_agg", -1))
            mlflow.log_metric("causalshd_agg", results.get("shd_agg", -1))
            mlflow.log_metric("causalnnz_agg", results.get("nnz_agg", -1))
            mlflow.log_metric("auroc_agg", results.get("auroc_agg", -1))

        else:
            raise ValueError("Invalid dimension of ground truth adjacency matrix")
    else:
        subgraph_idx = dataset.get_known_subgraph_mask_matrix()
        if len(adj_pred.shape) == 2:
            # If predicts single adjacency matrix
            results = edge_prediction_metrics(adj_ground_truth, adj_pred, adj_matrix_mask=subgraph_idx)
        elif len(adj_pred.shape) == 3:
            # If predicts multiple adjacency matrices (stacked)
            results = edge_prediction_metrics_multisample(adj_ground_truth, adj_pred, adj_matrix_mask=subgraph_idx)
        # Log metrics
        mlflow.log_metric("adjacency.recall", results["adjacency_recall"])
        mlflow.log_metric("adjacency.precision", results["adjacency_precision"])
        mlflow.log_metric("adjacency.f1", results["adjacency_fscore"], True)
        mlflow.log_metric("orientation.recall", results["orientation_recall"])
        mlflow.log_metric("orientation.precision", results["orientation_precision"])
        mlflow.log_metric("orientation.f1", results["orientation_fscore"], True)
        mlflow.log_metric("causal_accuracy", results["causal_accuracy"])
        mlflow.log_metric("causalshd", results["shd"])
        mlflow.log_metric("causalnnz", results["nnz"])

    # save adjacency matrix
    np.save(os.path.join(model.save_dir, "adj_matrices"), adj_pred, allow_pickle=True, fix_imports=True)

    # Save causality results to a file
    save_train_val_test_metrics(
        train_metrics={},
        val_metrics={},
        test_metrics=results,
        save_file=os.path.join(model.save_dir, "target_results_causality.json"),
    )


def eval_latent_confounded_causal_discovery(
    dataset: Union[CausalDataset, LatentConfoundedCausalDataset],
    model: IModelForCausalInference,
):
    if isinstance(dataset, LatentConfoundedCausalDataset):
        true_directed_adj = dataset.get_directed_adjacency_data_matrix()
        true_bidirected_adj = dataset.get_bidirected_adjacency_data_matrix()
    else:
        true_directed_adj = dataset.get_adjacency_data_matrix()
        true_bidirected_adj = np.zeros_like(true_directed_adj)

    if hasattr(model, "get_admg_matrices"):
        directed_adj, bidirected_adj = cast(Any, model).get_admg_matrices()
        directed_adj = directed_adj.astype(float).round()
        bidirected_adj = bidirected_adj.astype(float).round()
    elif isinstance(model, DDECI):
        return
    else:
        assert hasattr(model, "get_adj_matrix")
        directed_adj = model.get_adj_matrix().astype(float).round()
        bidirected_adj = np.zeros_like(directed_adj)

    # save matrices
    np.save(os.path.join(model.save_dir, "pred_directed_adj"), directed_adj, allow_pickle=True, fix_imports=True)
    np.save(os.path.join(model.save_dir, "pred_bidirected_adj"), bidirected_adj, allow_pickle=True, fix_imports=True)

    for adj_type, adj_pred, adj_ground_truth in zip(
        ["directed", "bidirected"], [directed_adj, bidirected_adj], [true_directed_adj, true_bidirected_adj]
    ):
        if len(adj_pred.shape) == 2:
            # If predicts single adjacency matrix
            results = edge_prediction_metrics(adj_ground_truth, adj_pred, adj_matrix_mask=None)
        elif len(adj_pred.shape) == 3:
            # If predicts multiple adjacency matrices (stacked)
            results = edge_prediction_metrics_multisample(adj_ground_truth, adj_pred, adj_matrix_mask=None)

        # Log metrics
        mlflow.log_metric(f"{adj_type}.adjacency.recall", results["adjacency_recall"])
        mlflow.log_metric(f"{adj_type}.adjacency.precision", results["adjacency_precision"])
        mlflow.log_metric(f"{adj_type}.adjacency.f1", results["adjacency_fscore"], True)
        mlflow.log_metric(f"{adj_type}.orientation.recall", results["orientation_recall"])
        mlflow.log_metric(f"{adj_type}.orientation.precision", results["orientation_precision"])
        mlflow.log_metric(f"{adj_type}.orientation.f1", results["orientation_fscore"], True)
        mlflow.log_metric(f"{adj_type}.causal_accuracy", results["causal_accuracy"])
        mlflow.log_metric(f"{adj_type}.causalshd", results["shd"])
        mlflow.log_metric(f"{adj_type}.causalnnz", results["nnz"])
        # Save causality results to a file
        save_train_val_test_metrics(
            train_metrics={},
            val_metrics={},
            test_metrics=results,
            save_file=os.path.join(model.save_dir, f"target_results_causality_{adj_type}.json"),
        )


def evaluate_treatment_effect_estimation(
    model: IModelForCausalInference,
    dataset: CausalDataset,
    logger: Logger,
    eval_likelihood: bool = True,
):
    """
    Evaluates treatment effect estimation made by `model` against ground-truth data.
    """

    process_dataset = isinstance(model, DECI)
    eval_treatment_effects(logger, dataset, model, eval_likelihood, process_dataset)

    if isinstance(model, IModelForCounterfactuals) and dataset.has_counterfactual_data:
        # only evaluate ITE if the model can generate counterfactuals and we have ground truth counterfactual data
        eval_individual_treatment_effects(dataset, model, process_dataset)


def eval_individual_treatment_effects(
    dataset: CausalDataset,
    model: IModelForCounterfactuals,
    process_dataset: bool = True,
) -> None:

    # Process test and intervention data in the same way that train data is processed
    if process_dataset:
        processed_dataset = model.data_processor.process_dataset(dataset)
    else:
        processed_dataset = dataset

    assert isinstance(processed_dataset, (TemporalDataset, CausalDataset))
    # Evaluate ITE
    get_ite_evaluation_results_partial = partial(
        get_ite_evaluation_results,
        model=model,
        counterfactual_datasets=processed_dataset.get_counterfactual_data(),
        variables=processed_dataset.variables,
        processed=process_dataset,
    )

    ite_rmse_metrics, ite_norm_rmse_metrics = get_ite_evaluation_results_partial(most_likely_graph=False)
    ite_rmse_most_likely_metrics, ite_norm_rmse_most_likely_metrics = get_ite_evaluation_results_partial(
        most_likely_graph=True, Ngraphs=1
    )
    # ITE
    counterfactual_metric_dict: Dict[str, Any] = {ALL_INTRVS: {}}
    _add_rmse_to_dict(counterfactual_metric_dict, ite_norm_rmse_metrics, "Normalised ITE RMSE")
    _add_rmse_to_dict(counterfactual_metric_dict, ite_rmse_metrics, "ITE RMSE")
    _add_rmse_to_dict(counterfactual_metric_dict, ite_norm_rmse_most_likely_metrics, "ML Normalised ITE RMSE")
    _add_rmse_to_dict(counterfactual_metric_dict, ite_rmse_most_likely_metrics, "ML ITE RMSE")

    # Log the metrics
    mlflow.log_metric("interventions.all.ite_rmse", ite_rmse_metrics.all)
    mlflow.log_metric("interventions.all.ML.ite_rmse", ite_norm_rmse_most_likely_metrics.all)
    mlflow.log_metric("ITE_rmse", ite_rmse_metrics.all)

    # Save counterfactual results to a file
    save_train_val_test_metrics(
        train_metrics={},
        val_metrics={},
        test_metrics=counterfactual_metric_dict,
        save_file=os.path.join(model.save_dir, "results_counterfactual.json"),
    )


def eval_treatment_effects(
    logger: Logger,
    dataset: CausalDataset,
    model: IModelForCausalInference,
    eval_likelihood: bool = True,
    process_dataset: bool = True,
) -> None:
    """
    Run treatment effect experiments: ATE RMSE and interventional distribution log-likelihood with graph marginalisation and most likely (ml) graph.
        Save results as json file and in metrics logger.
    Args:
        logger (`logging.Logger`): Instance of logger class to use.
        dataset: Dataset or SparseDataset object.
        model (IModelForInterventions): Model to use.
        name_prepend: Optional string that will be prepended to the json save name and logged metrics.
             This allows us to distinguish results from end2end models from results computed with downstream models (models that require graph as input, like DoWhy)
        eval_likelihood: Optional bool flag that will disable the log likelihood evaluation
        process_data: Whether to apply the data processor to the interventional data. This is done for internal models such as DECI, and not for DoWhy.

    This requires the model to implement methods sample() to sample from the model distribution with interventions
     and log_prob() to evaluate the density of test samples under interventions
    """
    # Process test and intervention data in the same way that train data is processed
    if process_dataset:
        processed_dataset = model.data_processor.process_dataset(dataset)
    else:
        processed_dataset = dataset
    test_data, _ = processed_dataset.test_data_and_mask

    assert isinstance(test_data, np.ndarray)
    assert isinstance(processed_dataset, (TemporalDataset, CausalDataset))
    # TODO: when scaling continuous variables in `process_dataset` we should call `revert_data` to evaluate RMSE in original space
    get_ate_rms_partial = partial(
        get_ate_rms,
        model=model,
        test_samples=test_data.astype(float),
        intervention_datasets=processed_dataset.get_intervention_data(),
        variables=processed_dataset.variables,
        processed=process_dataset,
    )

    ate_rmse_metrics, ate_norm_rmse_metrics = get_ate_rms_partial(most_likely_graph=False)
    ate_rmse_most_likely_metrics, ate_norm_rmse_most_likely_metrics = get_ate_rms_partial(most_likely_graph=True)

    if eval_likelihood:
        log_prob_treatments = get_treatment_data_logprob(model, processed_dataset.get_intervention_data())
        log_prob_most_likely_treatments = get_treatment_data_logprob(
            model, processed_dataset.get_intervention_data(), most_likely_graph=True
        )

        # Evaluate test log-prob only for models that support it
        if "do" not in model.name():
            base_testset_intervention = [
                InterventionData(
                    intervention_idxs=np.array([]), intervention_values=np.array([]), test_data=test_data.astype(float)
                )
            ]
            test_log_prob_treatments = get_treatment_data_logprob(model, base_testset_intervention)
        else:
            test_log_prob_treatments = None
    else:
        logger.info("Disable the log likelihood evaluation for this causal model")
        log_prob_treatments = None
        log_prob_most_likely_treatments = None
        test_log_prob_treatments = None

    # Create the metric dict
    metric_dict: Dict[str, Any] = {ALL_INTRVS: {}}

    # ATE
    _add_rmse_to_dict(metric_dict, ate_norm_rmse_metrics, "Normalised ATE RMSE")
    _add_rmse_to_dict(metric_dict, ate_rmse_metrics, "ATE RMSE")
    _add_rmse_to_dict(metric_dict, ate_norm_rmse_most_likely_metrics, "ML Normalised ATE RMSE")
    _add_rmse_to_dict(metric_dict, ate_rmse_most_likely_metrics, "ML ATE RMSE")

    _add_intervention_likelihoods_to_dict(metric_dict, log_prob_treatments, key_str="log prob {mean_or_std}")
    _add_intervention_likelihoods_to_dict(
        metric_dict, log_prob_most_likely_treatments, key_str="log prob {mean_or_std} ML Graph"
    )

    if test_log_prob_treatments is not None:
        metric_dict["test log prob mean"] = test_log_prob_treatments.all_mean
        metric_dict["test log prob std"] = test_log_prob_treatments.all_std

    # log the metrics
    mlflow.log_metric("interventions.all.ate_rmse", ate_rmse_metrics.all)
    mlflow.log_metric("interventions.all.ML.ate_rmse", ate_rmse_most_likely_metrics.all)

    _log_treatment(log_prob_treatments, logger_str="all.log_prob")
    _log_treatment(log_prob_most_likely_treatments, logger_str="all.ML.log_prob")
    _log_treatment(test_log_prob_treatments, logger_str="test.log_prob")

    # special metrics to log
    mlflow.log_metric("TE_rmse", ate_rmse_metrics.all)

    if log_prob_treatments is not None:
        mlflow.log_metric("TE_LL", log_prob_treatments.all_mean)
    if test_log_prob_treatments is not None:
        mlflow.log_metric("test_LL", test_log_prob_treatments.all_mean, True)

    # Save intervention results to a file
    save_train_val_test_metrics(
        train_metrics={},
        val_metrics={},
        test_metrics=metric_dict,
        save_file=os.path.join(model.save_dir, "results_interventions.json"),
    )


def _log_treatment(treatment_probs: Optional[TreatmentDataLogProb], logger_str: str):
    """Log treatment log probabilities to the metrics logger."""
    if treatment_probs is None:
        return

    mlflow.log_metric(f"interventions.{logger_str}_mean", treatment_probs.all_mean)
    mlflow.log_metric(f"interventions.{logger_str}_std", treatment_probs.all_std)


def _add_intervention_likelihoods_to_dict(
    metric_dict: Dict[str, Any], log_probs: Optional[TreatmentDataLogProb], key_str: str
):
    """Add treatment log probabilities to the metric dictionary."""
    if log_probs is None:
        return

    mean_str = key_str.format(mean_or_std="mean")
    std_str = key_str.format(mean_or_std="std")

    metric_dict[ALL_INTRVS][mean_str] = log_probs.all_mean
    metric_dict[ALL_INTRVS][std_str] = log_probs.all_std
    for n_int in range(len(log_probs.per_intervention_mean)):
        key = f"Intervention {n_int}"
        if key not in metric_dict.keys():
            metric_dict[key] = {}
        metric_dict[key][mean_str] = log_probs.per_intervention_mean[n_int]
        metric_dict[key][std_str] = log_probs.per_intervention_std[n_int]


def _add_rmse_to_dict(metric_dict: Dict[str, Any], rmse: Union[AteRMSEMetrics, IteRMSEMetrics], metric_name_key: str):
    """Add RMSE metrics to the metric dictionary."""
    metric_dict[ALL_INTRVS][metric_name_key] = rmse.all
    all_cols = "all columns"

    for n_int in range(rmse.n_interventions):
        int_key = f"Intervention {n_int}"
        if int_key not in metric_dict:
            metric_dict[int_key] = {}

        if all_cols not in metric_dict[int_key]:
            metric_dict[int_key][all_cols] = {}

        metric_dict[int_key][all_cols][metric_name_key] = rmse.across_groups[n_int]

        for dim in range(rmse.n_groups):
            column_key = f"Column {dim}"

            if column_key not in metric_dict[int_key]:
                metric_dict[int_key][column_key] = {}

            metric_dict[int_key][column_key][metric_name_key] = rmse.get_rmse(n_int, dim)
