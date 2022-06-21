import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import matplotlib

# pylint: disable=wrong-import-position
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix, issparse

from ..datasets.dataset import Dataset, SparseDataset
from ..datasets.variables import Variables
from ..models.imodel import IModelForImputation
from ..utils.metrics import compute_metrics
from .imputation_statistics_utils import ImputationStatistics as ImputeStats


def run_imputation(
    model: IModelForImputation,
    data: Union[np.ndarray, csr_matrix],
    observed_mask: Union[np.ndarray, csr_matrix],
    impute_config,
    vamp_prior_data,
):
    """
    Run missing value imputation, conditioned on the values in data marked as observed in observed_mask.

    Args:
        model: Model to use.
        data (shape (user_count, variable_count)): Data to perform imputation on. Should contain both the
            observed data and prediction target ground truths: these will be separated using masks.
        observed_mask (shape (user_count, variable_count)): Mask for observed values in data. 1 is observed, 0 is
            missing.
        impute_config (dictionary): Dictionary containing options for imputation.
        vamp_prior_data (tuple of numpy arrays): Tuple of (data, mask). Used for vamp prior samples.

    Returns:
        predictions (shape (user_count, variable_count): Fully imputed data array.
    """
    if vamp_prior_data is not None:
        raise NotImplementedError

    predictions = model.impute(data, observed_mask, impute_config, vamp_prior_data=None)
    return predictions


def run_imputation_with_stats(
    model: IModelForImputation,
    data: Union[np.ndarray, csr_matrix],
    observed_mask: Union[np.ndarray, csr_matrix],
    variables: Variables,
    impute_config,
    vamp_prior_data,
):
    """
    Run missing value imputation, conditioned on the values in data marked as observed in observed_mask, and compute
    statistics for the imputed values across multiple imputation samples.

    Args:
        model: Model to use.
        data (shape (user_count, variable_count)): Data to perform imputation on. Should contain both the
            observed data and prediction target ground truths: these will be separated using masks.
        observed_mask (shape (user_count, variable_count)): Mask for observed values in data. 1 is observed, 0 is
            missing.
        variables (Variables): Variables object containing variables metadata for the data.
        impute_config (dictionary): Dictionary containing options for inference.
        vamp_prior_data (tuple of numpy arrays): Tuple of (data, mask). Used for vamp prior samples.

    Returns:
        predictions (np array): Imputation values with shape (sample_count, user_count, feature_count).
        predictions_stats (dictionary): Imputation stats of {variable idx: stats_dictionary},
        where depending on the type of the variable (continuous/binary/categorical) the computed statistics type
        can be different.
    """
    if vamp_prior_data is not None:
        raise NotImplementedError

    # TODO: fix typing below. The code makes an assumption that we run it for PVAEBaseModel, as "average" keyword is used
    predictions = model.impute(data, observed_mask, impute_config, vamp_prior_data=None, average=False)  # type: ignore
    predictions_stats = ImputeStats.get_statistics(predictions, variables)

    return predictions, predictions_stats


def eval_imputation(
    dataset: Union[Dataset, SparseDataset],
    model: IModelForImputation,
    variables: Variables,
    split_type: str,
    vamp_prior_data,
    impute_config: Dict[str, Any],
    impute_train_data: bool,
    seed: int = 0,
) -> Tuple[
    Optional[Union[np.ndarray, csr_matrix]],
    Optional[Union[np.ndarray, csr_matrix]],
    Optional[np.ndarray],
    Dict[str, Any],
    Optional[Union[np.ndarray, csr_matrix]],
    Optional[Union[np.ndarray, csr_matrix]],
    Optional[np.ndarray],
    Dict[str, Any],
    Union[np.ndarray, csr_matrix],
    Union[np.ndarray, csr_matrix],
    np.ndarray,
    Dict[str, Any],
]:
    """
    Evaluate imputation performance of a model on the train, validation and test splits of a dataset.

    For split_type == "rows", the train/val/test splits are done by row, and we evaluate performance on each split by
    randomly observing some features in each row and using the remaining unseen features as prediction targets.

    For split_type == "elements", the train/val/test splits are assigned randomly across all rows and features, and we
    instead treat the training set elements as observed, and the validation and test set elements as prediction targets.
    Note that in this case, we will not return any metrics for the training set.

    Args:
        dataset: Dataset or SparseDataset object.
        model (IModel): Model to use.
        split_type (str): Whether test data is split by rows ('rows') or elements ('elements'). If 'elements', the test
            set values will be predicted conditioned on the training set values.
        vamp_prior_data (tuple of numpy arrays): Tuple of (data, mask). Used for vamp prior samples.
        impute_config (dictionary): Dictionary containing options for inference.
        objective_config (dictionary): Dictionary containing objective configuration parameters.
        impute_train_data (bool): Whether imputation should be run on training data. This can require much more memory.
        seed (int): Random seed to use when running eval.
    Returns:
        train_obs_mask: Boolean mask where 1 indicates features treated as observed during training set imputation and 0
            indicates those treated as unobserved.
        train_target_mask: Boolean mask where 1 indicates features treated as prediction targets during training set
            imputation and 0 indicates those not treated as prediction targets.
        train_imputations: If split_type == "rows", imputed data for the training set rows. Otherwise None.
        train_metrics: If split_type == "rows", metrics computed on the training set rows. Otherwise None.
        val_obs_mask: Boolean mask where 1 indicates features treated as observed during validation set imputation and 0
            indicates those treated as unobserved.
        val_target_mask: Boolean mask where 1 indicates features treated as prediction targets during validation set
            imputation and 0 indicates those not treated as prediction targets.
        val_imputations: If split_type == "rows", imputed data for the validation set rows. Otherwise None.
        val_metrics: Metrics computed on the validation set.
        test_obs_mask: Boolean mask where 1 indicates features treated as observed during test set imputation and 0
            indicates those treated as unobserved.
        test_target_mask: Boolean mask where 1 indicates features treated as prediction targets during test set
            imputation and 0 indicates those not treated as prediction targets.
        test_imputations: If split_type == "rows", imputed data for the test set rows. If split_type == "elements",
            imputed data for the whole dataset conditioned on the training set.
        test_metrics: Metrics computed on the test set.
    """
    train_data, train_mask = dataset.train_data_and_mask
    val_data, val_mask = dataset.val_data_and_mask
    test_data, test_mask = dataset.test_data_and_mask
    # Evaluate imputation on training data. Note we can't do this for elementwise data splits, since here we always
    # condition the imputation on the training set data.
    if split_type == "rows" and impute_train_data:
        train_obs_mask, train_target_mask = split_mask(train_mask, seed=seed)
        train_imputations = run_imputation(model, train_data, train_obs_mask, impute_config, vamp_prior_data)
        train_metrics = compute_metrics(train_imputations, train_data, train_target_mask, variables)
    else:
        train_obs_mask, train_target_mask, train_imputations = None, None, None
        train_metrics = {}

    # Evaluate imputation on test data
    if split_type == "elements":
        assert test_data is not None
        assert test_mask is not None
        test_obs_mask, test_target_mask = train_mask, test_mask
        # need to take into account the validation set,
        # as the test imputation result will be used in validation set evaluation.
        if val_mask is not None:
            rows_with_targets = np.ravel(np.sum(val_mask + test_target_mask, 1)) != 0
        else:
            rows_with_targets = np.ravel(np.sum(test_target_mask, 1)) != 0
        test_imputations = run_imputation(
            model, train_data[rows_with_targets, :], test_obs_mask[rows_with_targets, :], impute_config, vamp_prior_data
        )
        test_metrics = compute_metrics(
            test_imputations, test_data[rows_with_targets, :], test_target_mask[rows_with_targets, :], variables
        )

    else:
        test_obs_mask, test_target_mask = split_mask(test_mask, seed=seed)
        test_imputations = run_imputation(model, test_data, test_obs_mask, impute_config, vamp_prior_data)
        test_metrics = compute_metrics(test_imputations, test_data, test_target_mask, variables)

    # Evaluate imputation on validation data (if present).
    if val_data is not None and val_mask is not None:
        if split_type == "elements":
            val_obs_mask, val_target_mask = train_mask, val_mask
            rows_with_targets = np.ravel(np.sum(val_mask + test_target_mask, 1)) != 0
            rows_with_targets_val = np.ravel(np.sum(val_target_mask, 1)) != 0
            # For elementwise split, both val and test imputations are conditioned on training set so we can just reuse.
            # We compute validation metrics after test metrics for this reason, since we can always guarantee we'll do
            # imputation for test data, but not for validation data.
            val_imputations = test_imputations
            val_metrics = compute_metrics(
                val_imputations[rows_with_targets_val[rows_with_targets], :],
                val_data[rows_with_targets_val, :],
                val_target_mask[rows_with_targets_val, :],
                variables,
            )

        else:
            val_obs_mask, val_target_mask = split_mask(val_mask, seed=seed)
            val_imputations = run_imputation(model, val_data, val_obs_mask, impute_config, vamp_prior_data)
            val_metrics = compute_metrics(val_imputations, val_data, val_target_mask, variables)
    else:
        val_obs_mask, val_target_mask, val_imputations = None, None, None
        val_metrics = {}
    return (
        train_obs_mask,
        train_target_mask,
        train_imputations,
        train_metrics,
        val_obs_mask,
        val_target_mask,
        val_imputations,
        val_metrics,
        test_obs_mask,
        test_target_mask,
        test_imputations,
        test_metrics,
    )


def split_mask(mask, target_prob: float = 0.3, seed: Optional[int] = None):
    """
    Split a mask into an observed_mask and target_mask, by randomly marking some features that were previously
    observed as prediction targets with probability 'target_prob'. The outputs observed_mask and target_mask should
    produce the input mask when summed.

    Args:
        mask (numpy array of shape (user_count, variable_count)): the original mask
        drop_features_prob (float): probability of a feature being selected as a prediction target
        seed (int): seed for the random number generation

    Returns:
        observed_mask (shape (user_count, variable_count): mask where 1 indicates an observed value.
        target_mask (shape (user_count, variable_count)): mask where 1 indicates a prediction target.
    """
    sparse = issparse(mask)
    if seed is not None:
        np.random.seed(seed)

    assert 0 <= target_prob <= 1, "Probability must be in interval [0,1]."

    observed_mask = mask.copy()
    if sparse:
        target_mask = csr_matrix(mask.shape, dtype=bool)
    else:
        target_mask = np.zeros_like(mask, dtype=bool)
    all_prediction_targets = []
    for row_idx in range(mask.shape[0]):
        # Can't iterate through rows directly using `for row in mask` as sparse matrices don't support this
        mask_row = mask[row_idx]
        if sparse:
            # Indexing into row in sparse array doesn't drop row dimension, so need 2nd coordinate here.
            obs_features = mask_row.nonzero()[1]
        else:
            obs_features = np.where(mask_row == 1)[0]
        prob_vector_mask = np.random.rand(obs_features.shape[0]) < target_prob
        prediction_targets = obs_features[prob_vector_mask]
        observed_mask[row_idx, prediction_targets] = 0
        target_mask[row_idx, prediction_targets] = 1
        all_prediction_targets.append(target_mask)

    return observed_mask, target_mask


def impute_targets_only(model: IModelForImputation, test_data, test_mask, impute_config, vamp_prior_data):
    """
    Run missing value imputation for target variables only, using all available data.

    Args:
        model: Model to use.
        test_data (numpy array of shape (user_count, variable_count)): Data to run active learning on.
        test_mask (numpy array of shape (user_count, variable_count)): 1 is observed, 0 is missing.
        impute_config (dictionary): Dictionary containing options for inference.
        vamp_prior_data (tuple of numpy arrays): Tuple of (data, mask). Used for vamp prior samples.

    Returns:
        predictions: Imputation values with shape (user_count, feature_count).
    """

    test_mask = test_mask.copy()
    target_idxs = [idx for idx, var in enumerate(model.variables) if not var.query]
    # Mask out targets in copied array
    test_mask[:, target_idxs] = 0

    predictions = model.impute(test_data, test_mask, impute_config, vamp_prior_data=vamp_prior_data)
    return predictions


def plot_pairwise_comparison(data, variables, filename_suffix, save_dir):
    logger = logging.getLogger()
    col_names = [var.name for var in variables]
    data_frame = pd.DataFrame(data=data, columns=col_names)

    logger.info("Generating pair plot... (this may take a while)")
    sns.pairplot(data_frame)

    save_path = os.path.join(save_dir, f"pairwise_{filename_suffix}.png")
    plt.savefig(save_path, format="png", dpi=200, bbox_inches="tight")
    logger.info("Saved plot to %s", save_path)
