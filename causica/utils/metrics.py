import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import torch
from scipy.sparse import csr_matrix, issparse, spmatrix
from sklearn import metrics as sk_metrics
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from ..datasets.variables import Variables
from ..models.imodel import IModelForObjective
from ..utils.helper_functions import maintain_random_state
from ..utils.io_utils import format_dict_for_console, save_json

logger = logging.getLogger(__name__)


def as_str(val, fmt):
    if isinstance(val, tuple) and len(val) == 2:
        return (fmt + " ± " + fmt) % (val[0], val[1])
    else:
        return fmt % val


def as_float(val):
    if isinstance(val, tuple):
        return tuple(float(v) for v in val)
    else:
        return float(val)


def compute_and_save_metrics(imputed_values, ground_truth, target_mask, variables, save_file) -> Dict:
    # Imputation metrics for targets specified in target_mask
    metrics = compute_metrics(imputed_values, ground_truth, target_mask, variables)
    save_metrics_and_confusion_plot(metrics, save_file)
    return metrics


def save_metrics_and_confusion_plot(metrics: Dict, save_file: str):
    """
    For imputation metrics as returned by compute_metrics
    Write to JSON, log to console, and make a confusion plot
    """

    logger.info("Metrics: %s", format_dict_for_console(metrics))
    save_json(metrics, save_file)
    save_confusion_plot(metrics, os.path.dirname(save_file))


def save_train_val_test_metrics(
    train_metrics: Dict[str, Dict[str, Any]],
    val_metrics: Dict[str, Dict[str, Any]],
    test_metrics: Dict[str, Dict[str, Any]],
    save_file: str,
):
    """
    Save the metrics dictionaries corresponding to train/val/test data in one file.
    """
    metrics = {"train_data": train_metrics, "val_data": val_metrics, "test_data": test_metrics}
    save_json(metrics, save_file)


def compute_metrics(imputed_values, ground_truth, target_mask, variables: Variables) -> Dict[str, Any]:
    """
    Compute imputation metrics.

    Args:
        imputed_values: Imputation values with shape (user_count, feature_count).
        ground_truth: Ground truth values with shape (user_count, feature_count).
        target_mask: Boolean mask indicating prediction targets, where 1 is a target (user_count, feature_count).
        variables: Variables object.

    Returns:
        Dictionary of metric results {var: {metric: value}}

    """
    assert target_mask.dtype == bool  # Ensure mask is bool so that using ~mask in other functions behaves as expected.

    if issparse(ground_truth):
        # Need to do computations with dense imputed values array, so if ground truth is sparse convert to dense array
        # here.
        ground_truth = ground_truth.toarray()

    if isinstance(target_mask, csr_matrix):
        # Metrics computed per column so it is faster to convert sparse masks to a CSC matrix here for fast indexing
        target_mask = target_mask.tocsc()

    # remove auxiliary variables when computing metrics
    if variables.has_auxiliary:
        ground_truth = ground_truth[:, 0 : variables.num_processed_non_aux_cols]
        target_mask = target_mask[:, 0 : variables.num_processed_non_aux_cols]
        variables = variables.subset(list(range(0, variables.num_unprocessed_non_aux_cols)))

    # Save results for easy access later.
    results = {}
    for idx, var in enumerate(variables):
        metric_dict = get_metric(variables, imputed_values, ground_truth, target_mask, [idx])
        results[var.name] = metric_dict

    metric_dict = {}
    # Compute area under the PR and the ROC curves for the binary variables
    if any(x.type_ == "binary" for x in variables):
        AUROC, AUPR = get_area_under_ROC_PR(imputed_values, ground_truth, target_mask, variables)
        metric_dict["AUROC"] = 0 if AUROC is None else AUROC
        metric_dict["AUPR"] = 0 if AUPR is None else AUPR
    # Calculate summary RMSE.
    # This may not be the best for some features, but is useful as a summary metric.
    all_rmse = get_rmse(imputed_values, ground_truth, target_mask, variables, normalise=False)
    all_rmse_normalised = get_rmse(imputed_values, ground_truth, target_mask, variables, normalise=True)
    metric_dict["Normalised RMSE"] = all_rmse_normalised
    metric_dict["RMSE"] = all_rmse

    # Compute aggregated accuracy across all categorical variables
    if any(x.type_ in {"categorical", "binary"} for x in variables):
        metric_dict["Accuracy"] = get_aggregated_accuracy(imputed_values, ground_truth, target_mask, variables)

    results["all"] = metric_dict

    return results


def save_confusion_plot(metrics: Dict[str, Dict[str, Any]], save_dir: str) -> None:
    """
    Plot confusion matrix and save image in {save_dir}/confusion_matrix.pdf

    Args:
        metrics
        save_dir (str): directory to save the image in
    """
    try:
        confusion = metrics["all"]["Confusion matrix"]
    except KeyError:
        # There is no confusion matrix to plot. Expected if there are no binary variables.
        return
    if isinstance(confusion, dict):
        # These are aggregated metrics
        confusion_matrix = confusion["mean"]
    else:
        confusion_matrix = confusion
    if confusion_matrix is None:
        return

    plt.figure(figsize=(4, 4))
    sn.heatmap(
        confusion_matrix,
        annot=True,
        cmap="YlGnBu",
        xticklabels=["Real 0", "Real 1"],
        yticklabels=["Predicted 0", "Predicted 1"],
    )
    save_cm = os.path.join(save_dir, "confusion_matrix.pdf")
    plt.savefig(save_cm, bbox_inches="tight")


def compute_target_metrics(imputed_values, ground_truth, variables) -> Dict[str, Dict[str, float]]:
    """

    Compute metrics for active learning target variable imputation.

    Args:
        imputed_values: Imputation values with shape (user_count, feature_count).
        ground_truth: Ground truth values with shape (user_count, feature_count).
        variables: List of variables.

    Returns:
        Dict of metrics {var: {metric: value}}
    """
    assert imputed_values.ndim == 2

    if issparse(ground_truth):
        # Need to do computations with dense imputed values array, so if ground truth is sparse convert to dense array
        # here.
        ground_truth = ground_truth.toarray()

    if variables.has_auxiliary:
        ground_truth = ground_truth[:, 0 : variables.num_processed_non_aux_cols]
        variables = variables.subset(list(range(0, variables.num_unprocessed_non_aux_cols)))

    # Create mask
    # Set active learning targets to be 1
    target_idxs = [idx for idx, var in enumerate(variables) if var.target]
    target_mask = np.zeros_like(imputed_values, dtype=bool)
    target_mask[:, target_idxs] = 1

    results = {}
    for idx, variable in enumerate(variables):
        if variable.query:
            continue

        metric_dict = get_metric(variables, imputed_values, ground_truth, target_mask, [idx])
        results[variable.name] = metric_dict

    return results


def get_metric(
    variables: Variables,
    imputed_values: np.ndarray,
    ground_truth: np.ndarray,
    target_mask: Union[np.ndarray, csr_matrix],
    idxs: List[int],
) -> Dict[str, float]:
    """
    Get the value of a comparative metric for the given variables.

    Args:
        variables: Variables object
        imputed_values: Imputed values, shape (user_count, feature_count).
        ground_truth: Ground truth values, shape (user_count, feature_count).
        mask: Boolean mask indicating prediction targets, where 1 is a target. Shape (user_count, feature_count).
        idxs: Indices of variables to get the metric for.

    Return:
        Dict of {metric name: value}
    """
    assert imputed_values.ndim == 2
    assert target_mask.ndim == 2
    assert target_mask.dtype == bool

    # TODO move this function to a utils module as it's defined in a few places
    def flatten(lists):
        return [i for sublist in lists for i in sublist]

    cols = flatten([variables.unprocessed_non_aux_cols[i] for i in idxs])

    # if not isinstance(idxs, list):
    #     idxs = [idxs]
    if isinstance(target_mask, spmatrix):
        target_mask = target_mask.tocsc()

    # TODO: doing this here means variables won't line up in any of below. Better to pass refs to all
    # data and do subsetting in individual functions?
    # Get columns we care about for imputed values, mask and ground truth.
    imputed_values = imputed_values[:, cols]  # Shape (user_count, idx_count)
    target_mask = target_mask[:, cols]  # Shape (user_count, idx_count)
    ground_truth = ground_truth[:, cols]  # Shape (user_count, idx_count)

    unique_variables_types = list(set(variables[i].type_ for i in idxs))
    if len(unique_variables_types) > 1:
        raise ValueError("All of the variables should be of the same type")
    variables_type: str = unique_variables_types[0]

    subset_variables = variables.subset(idxs)

    if variables_type == "continuous":
        normalised_rmse = get_rmse(imputed_values, ground_truth, target_mask, subset_variables, normalise=True)
        rmse = get_rmse(imputed_values, ground_truth, target_mask, subset_variables, normalise=False)
        return {"Normalised RMSE": normalised_rmse, "RMSE": rmse}
    elif variables_type == "binary":
        # Round imputed_values to nearest value before comparing with ground truth.
        rounded_values = imputed_values.astype(float).round()
        return {
            "Fraction Incorrectly classified": get_fraction_incorrectly_classified(
                rounded_values, ground_truth, target_mask
            )
        }
    elif variables_type == "categorical":
        return {
            "Fraction Incorrectly classified": get_fraction_incorrectly_classified(
                imputed_values, ground_truth, target_mask
            )
        }
    elif variables_type == "text":
        return {"Mock text metrics": float("nan")}  # TODO #18598: Add metrics for text variable
    else:
        raise ValueError(
            f"Incorrect variable type. Expected one of continuous, binary or categorical. Was {variables_type}."
        )


def get_fraction_incorrectly_classified(imputed_values, ground_truth, target_mask):
    """
    Get fraction of categorical values that are wrongly imputed.
    Args:
        imputed_values: Imputed values, shape (user_count, feature_count).
        ground_truth: Ground truth values, shape (user_count, feature_count).
        target_mask: Boolean ask indicating prediction targets, where 1 is a target. Shape (user_count, feature_count).

    Returns:
        Fraction incorrectly classified. Value is between 0 and 1, where 1 is all mis-classified.
    """
    assert imputed_values.ndim == 2
    assert target_mask.ndim == 2
    assert target_mask.dtype == bool
    errs = ~(ground_truth == imputed_values)  # Boolean - 0 where correctly classified, 1 where incorrectly classified.
    # Only consider elements marked as targets
    target_errs = errs[target_mask.nonzero()]
    return target_errs.mean()


def get_rmse(imputed_values, ground_truth, target_mask, variables: Variables, normalise: bool, batch_size=10000):
    """
    Get RMSE (Root Mean Squared Error) between imputed and ground truth data.

    Args:
        imputed_values: Imputation values with shape (user_count, feature_count).
        ground_truth: Expected values to be compared with imputed_values. Shape (user_count, feature_count).
        target_mask: Boolean mask indicating prediction targets, where 1 is a target. Shape (user_count, feature_count).
        variables: Variable object for each feature to use.
        normalise: Whether or not to normalise RMSE.

    Returns:
        RMSE mean and stddev across seeds.
    """
    assert imputed_values.ndim == 2
    assert target_mask.ndim == 2
    assert target_mask.dtype == bool

    total_sq_err = 0
    total_num_el = 0
    num_rows = imputed_values.shape[0]

    non_text_indices = ~np.in1d(range(ground_truth.shape[1]), variables.text_idxs)
    for start_idx in range(0, num_rows, batch_size):
        stop_idx = min(start_idx + batch_size, num_rows)
        # Copy the imputed_values variable to avoid it being modified
        imputed_values_batch = imputed_values[start_idx:stop_idx].copy()
        ground_truth_batch = ground_truth[start_idx:stop_idx].copy()
        target_mask_batch = target_mask[start_idx:stop_idx]

        if normalise:
            # Normalise values between 0 and 1.
            # TODO 18375: can we avoid the repeated (un)normalization of data before/during this function or at least
            # share the normalization logic in both places?
            lowers = np.array([var.lower for var in variables if var.type_ != "text"])
            uppers = np.array([var.upper for var in variables if var.type_ != "text"])
            imputed_values_batch[:, non_text_indices] = (imputed_values_batch[:, non_text_indices] - lowers) / (
                uppers - lowers
            )
            ground_truth_batch[:, non_text_indices] = (ground_truth_batch[:, non_text_indices] - lowers) / (
                uppers - lowers
            )

        # Convert binary predictions to 0/1
        imputed_values_batch[:, variables.binary_idxs] = (imputed_values_batch[:, variables.binary_idxs] >= 0.5).astype(
            float
        )

        # Calculate squared errors (ignoring text variables).
        sq_errs_batch = np.zeros_like(ground_truth_batch, dtype=float)
        sq_errs_batch[:, non_text_indices] = (
            ground_truth_batch[:, non_text_indices] - imputed_values_batch[:, non_text_indices]
        ) ** 2  # Shape (batch_size, num_features)

        # For categorical columns, assign a squared error of 1 when incorrect
        for cat_idx in variables.categorical_idxs:
            col_nonzero_elements = sq_errs_batch[:, cat_idx].nonzero()
            sq_errs_batch[col_nonzero_elements, cat_idx] = 1

        # Only consider elements marked as targets when computing the metric
        total_sq_err += sq_errs_batch[target_mask_batch.nonzero()].sum()
        total_num_el += target_mask_batch.sum()

    return np.sqrt(total_sq_err / total_num_el)


def get_aggregated_accuracy(
    imputed_values: np.ndarray,
    ground_truth: np.ndarray,
    target_mask: np.ndarray,
    variables: Variables,
) -> float:
    """
    Get accuracy calculated across all discrete variables in the dataset.

    Args:
        imputed_values: Imputation values with shape (user_count, feature_count).
        ground_truth: Expected values to be compared with imputed_values. Shape (user_count, feature_count).
        target_mask: Mask indicating prediction targets, where 1 is a target. Shape (user_count, feature_count).
        variables: Variable object for each feature to use.

    Returns:
        acc: accuracy or mean and stddev of accuracy across seeds.
    """
    assert imputed_values.ndim == 2
    assert target_mask.ndim == 2
    discrete_var_idxs = variables.discrete_idxs
    discrete_imputed_values = imputed_values[:, discrete_var_idxs].astype(float)

    # Currently binary predictions will be real valued in [0,1], so round to 0 or 1.
    discrete_imputed_values = discrete_imputed_values.round()
    discrete_ground_truth = ground_truth[:, discrete_var_idxs]
    discrete_target_mask = target_mask[:, discrete_var_idxs]

    fraction_incorrectly_classified = get_fraction_incorrectly_classified(
        discrete_imputed_values, discrete_ground_truth, discrete_target_mask
    )
    acc = 1 - fraction_incorrectly_classified
    return acc


def get_aggregated_binary_confusion_matrix(imputed_values, ground_truth, target_mask, variables) -> List[List[float]]:
    """
    Get the confusion matrix calculated across all binary variables in the dataset.

    Args:
        imputed_values: Imputation values with shape (user_count, feature_count).
        ground_truth: Expected values to be compared with imputed_values. Shape (user_count, feature_count).
        target_mask: Mask indicating prediction targets, where 1 is a target. Shape (user_count, feature_count).
        variables: Variable object for each feature to use.

    Returns:
        cm: confusion matrix i.e. [[class0_accuracy,1-class1_accuracy],
                    [1-class0_accuracy,class1_accuracy]]
    """
    assert imputed_values.ndim == 2
    assert target_mask.ndim == 2
    assert target_mask.dtype == bool

    binary_imputed_values = imputed_values[:, variables.binary_idxs]

    # Currently binary predictions will be real valued in [0,1], so round to 0 or 1.
    binary_imputed_values = binary_imputed_values.astype(float).round()
    binary_ground_truth = ground_truth[:, variables.binary_idxs]
    binary_target_mask = target_mask[:, variables.binary_idxs]

    # Getting the accuracy for each one of the classes
    class0_count = (binary_ground_truth == 0)[binary_target_mask.nonzero()].sum()
    class1_count = (binary_ground_truth == 1)[binary_target_mask.nonzero()].sum()
    class0_correct_count = ((binary_ground_truth == 0) & (binary_imputed_values == 0))[
        binary_target_mask.nonzero()
    ].sum()
    class1_correct_count = ((binary_ground_truth == 1) & (binary_imputed_values == 1))[
        binary_target_mask.nonzero()
    ].sum()

    class0_accuracy = class0_correct_count / class0_count
    class1_accuracy = class1_correct_count / class1_count

    cm = [
        [class0_accuracy, 1 - class1_accuracy],
        [1 - class0_accuracy, class1_accuracy],
    ]
    # np.array(..).tolist() converts masked values to np.nan and makes result JSON-serialisable
    return np.array(cm).tolist()


def get_area_under_ROC_PR(
    imputed_values, ground_truth, target_mask, variables
) -> Tuple[Optional[float], Optional[float]]:
    """
    Get the area under the ROC and the PR curves for the binary variables.

    Args:
        imputed_values: Imputation values with shape (user_count, feature_count).
        ground_truth: Expected values to be compared with imputed_values. Shape (user_count, feature_count).
        target_mask: Mask indicating prediction targets, where 1 is a target. Shape (user_count, feature_count).
        variables: Variable object for each feature to use.

    Returns:
        (AUROC,AUPR) where AUROC is the area under the ROC curve and AUPR is the area under the PR curve.
    """
    assert imputed_values.ndim == 2
    assert target_mask.ndim == 2

    binary_imputed_values = imputed_values[:, variables.binary_idxs].astype(float)
    binary_target_mask = target_mask[:, variables.binary_idxs]
    binary_gt = ground_truth[:, variables.binary_idxs].astype(float)

    if binary_target_mask.sum() > 0:
        pred = binary_imputed_values[binary_target_mask.nonzero()]
        y = binary_gt[binary_target_mask.nonzero()]

        fpr, tpr, _ = sk_metrics.roc_curve(y, pred)
        AUROC = sk_metrics.auc(fpr, tpr)
        precision, recall, _ = sk_metrics.precision_recall_curve(y, pred)
        AUPR = sk_metrics.auc(recall, precision)

        return AUROC, AUPR
    else:
        return None, None


def create_latent_distribution_plots(
    model: IModelForObjective, dataloader: DataLoader, output_dir: str, epoch: int, num_points_plot: int
) -> None:
    """
    Create latent distribution plots using samples from given data and mask.

    Three plots are created:
       - SNR histogram of latent |mean|/std in dB.
       - tSNE plot of latent mean and log-variance, with points coloured by target value,
         or just the first variable if there is no target variable.
       - Scatter plot of latent mean vs. log-variance, with points coloured by latent
         dimension.
    Epoch number is included as a suffix in the filename for each plot.

    Args:
        model (IModelForObjective): Any model which implements 'encode'.
        dataloader: DataLoader instance supplying data and mask inputs.
        output_dir: Training output dir. Plots will be saved at e.g.
           {output_dir}/latent_distribution_plots/SNR_histogram{epoch}.png
        epoch (int): Number of epochs of training that have happened.
        num_points_plot: Number of points to plot. Will be rounded up to a whole number of batches from the dataloader.
    """
    with torch.no_grad():
        with maintain_random_state():  # Otherwise, this advances random number generator
            plots_dir = os.path.join(output_dir, "latent_distribution_plots")
            os.makedirs(plots_dir, exist_ok=True)
            target_idx, target_variable = next(
                ((idx, v) for idx, v in enumerate(model.variables) if not v.query), (0, model.variables[0])
            )
            device = model.get_device()
            num_points = 0
            z_and_target = []
            for data, mask in dataloader:
                z = model.encode(data.to(device), mask.to(device))
                target = model.variables.get_var_cols_from_data(target_idx, data)
                z_and_target.append((z, target))
                num_points += data.shape[0]
                if num_points >= num_points_plot:
                    break
            z_mean = np.concatenate([x[0][0].detach().cpu().numpy() for x in z_and_target], axis=0)
            z_logvar = np.concatenate([x[0][1].detach().cpu().numpy() for x in z_and_target], axis=0)
            target_values = np.concatenate([x[1].detach().cpu().numpy() for x in z_and_target], axis=0)

            # Scatter plot of mean vs variance
            plt.figure()
            plt.plot(z_mean, z_logvar, ".")
            plt.xlabel("mean")
            plt.ylabel("log(variance)")
            plt.title(f"Latent distribution parameters, coloured by latent dimension. Epoch {epoch}")
            plt.savefig(os.path.join(plots_dir, f"mean_vs_logvar{epoch}.png"))
            plt.clf()

            # SNR histogram of |mean| divided by standard deviation
            eps = 1e-20  # Truncate logvar to -20
            plt.hist(
                np.log10(np.maximum(eps, z_mean.reshape(-1) ** 2) / np.maximum(eps, np.exp(z_logvar.reshape(-1)))) * 5,
                20,
            )
            plt.title(f"Latent signal-to-noise ratio. Epoch {epoch}")
            plt.xlabel("|mean|/std in dB")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(plots_dir, f"SNR_histogram{epoch}.png"))
            tsne_embedded = TSNE(n_components=2).fit_transform(z_mean)
            plt.clf()

            # t-SNE plot, coloured by target variable if there is one, or just the first variable if not
            plt.scatter(tsne_embedded[:, 0], tsne_embedded[:, 1], s=5, c=target_values[:, 0])
            plt.title(
                f"t-SNE plot of latent distribution parameters, coloured by {target_variable.name}. Epoch {epoch}"
            )
            plt.savefig(os.path.join(plots_dir, f"tSNE{epoch}.png"))
            plt.close()
