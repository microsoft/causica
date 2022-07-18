from typing import Optional, Tuple

import torch
import torch.distributions as tdist
from torch.distributions.distribution import Distribution

from ..datasets.variables import Variables


def get_input_and_scoring_masks(
    mask: torch.Tensor, *, max_p_train_dropout: float, score_imputation: bool, score_reconstruction: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random dropout to an unprocessed mask, and calculate output positions to be included in the training loss.

    Args:
        mask: unprocessed mask indicating which variables are available for training. 1 indicates observed, 0 indicates
            unobserved.
        max_p_train_dropout: max proportion of columns to mask in each row.
        score_imputation: If true, the scoring mask has 1.0 for entries that are present in the data but masked in the
            input to the model.
        score_reconstruction: if true, the scoring mask has 1.0 for entries that are unmasked in the model input.

    Returns:
        A tuple (input_mask, scoring_mask), where input_mask is is the unprocessed mask to be applied before passing
        data to the model for reconstruction/imputation. scoring_mask (also, unprocessed mask) indicates which entries
        in the output should be included when calculating negative log-likelihood loss.
    """

    if max_p_train_dropout > 0:
        p_missing = torch.rand(mask.shape[0], 1) * max_p_train_dropout
        input_mask = mask * torch.bernoulli(1.0 - p_missing.expand_as(mask)).to(mask.dtype).to(mask.device)
    else:
        input_mask = mask
    if score_reconstruction:
        if score_imputation:
            # Score both reconstruction and imputation
            scoring_mask = mask
        else:
            # Only score reconstruction
            scoring_mask = input_mask
    else:
        # Only score imputation
        scoring_mask = mask - input_mask
    return input_mask, scoring_mask


def bernoulli_negative_log_likelihood(
    targets: torch.Tensor,
    mean: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sum_type: Optional[str] = "all",
):
    """
    This function comptutes negative log likelihood for a Bernoulli distribution given some data.

    Args:
        targets: Ground truth values.
        mean: Predicted mean values, output from decoder.
        mask: Missingness mask. 1 is observed, 0 is missing. Defaults to None, in which cases assumed all observed.
        sum_type: How to sum result. None will return the entire array, 'cols' will sum per variable,'all' will sum all
            elements.

    Returns:
        nll: Negative log likelihood summed as per `sum_type`. torch.Tensor of shape (batch_size, num_vars)
        if `sum_type=='all'`, shape (1, num_vars) if `sum_type=='cols'` or a scalar if `sum_type is None`.
    """
    assert sum_type in [None, "cols", "all"]
    predicted_dist = tdist.Bernoulli(probs=mean)
    nll = get_nll_from_dist(predicted_dist, targets, mask, sum_type=sum_type)
    return nll


def gaussian_negative_log_likelihood(
    targets: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sum_type: Optional[str] = "all",
):
    """
    Compute the negative log likelihood for a Gaussian distribution given some data.

    Args:
        targets: Ground truth values.
        mean: Predicted mean values, output from decoder.
        logvar: Predicted logvar values, output from decoder.
        mask: Missingness mask. 1 is observed, 0 is missing. Defaults to None, in which cases assumed all observed.
        sum_type: How to sum result. None will return the entire array, 'cols' will sum per variable,'all' will sum all
            elements.

    Returns:
        nll: Negative log likelihood summed as per `sum_type`. torch.Tensor of shape (batch_size, num_vars)
        if `sum_type=='all'`, shape (1, num_vars) if `sum_type=='cols'` or a scalar if `sum_type is None`.
    """
    assert sum_type in [None, "cols", "all"]
    predicted_dist = tdist.Normal(loc=mean, scale=logvar.exp().sqrt())
    return get_nll_from_dist(predicted_dist, targets, mask, sum_type=sum_type)


def categorical_negative_log_likelihood(
    targets: torch.Tensor,
    mean: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sum_type: Optional[str] = "all",
):
    """
    This function comptutes negative log likelihood for a categorical distribution given some data.

    Args:
        targets: Ground truth values.
        mean: Predicted mean values, output from decoder.
        mask: Missingness mask. 1 is observed, 0 is missing. Defaults to None, in which cases assumed all observed.
        sum_type: How to sum result. None will return the entire array, 'cols' will sum per variable,'all' will sum all
            elements.

    Returns:
        nll: Negative log likelihood summed as per `sum_type`. torch.Tensor of shape (batch_size, num_vars)
        if `sum_type=='all'`, shape (1, num_vars) if `sum_type=='cols'` or a scalar if `sum_type is None`.
    """
    assert sum_type in [None, "cols", "all"]
    predicted_dist = tdist.OneHotCategorical(probs=mean)
    if mask is not None:
        mask = mask[:, 0]  # We only need one value for the mask, so don't use the extra dim.
    nll = get_nll_from_dist(predicted_dist, targets, mask, sum_type=sum_type)
    return nll


def get_nll_from_dist(
    dist: Distribution,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sum_type: Optional[str] = "all",
):
    assert sum_type in [None, "cols", "all"]
    log_prob = dist.log_prob(targets)
    if mask is not None:
        log_prob *= mask
    if sum_type == "all":
        nll = -1 * log_prob.sum()
    elif sum_type == "cols":
        nll = -1 * log_prob.sum(axis=0)
    else:
        nll = -1 * log_prob
    return nll


def kl_divergence(
    z1: Tuple[torch.Tensor, torch.Tensor],
    z2: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    mean1, logvar1 = z1

    if z2 is not None:
        mean2, logvar2 = z2
    else:
        mean2 = torch.zeros_like(mean1)
        logvar2 = torch.zeros_like(logvar1)

    sigma1 = logvar1.exp().sqrt()
    sigma2 = logvar2.exp().sqrt()

    normal1 = tdist.Normal(mean1, sigma1)
    normal2 = tdist.Normal(mean2, sigma2)

    kld = tdist.kl_divergence(normal1, normal2)
    kld = kld.sum(axis=1)
    return kld


def negative_log_likelihood(
    data: torch.Tensor,
    decoder_mean: torch.Tensor,
    decoder_logvar: torch.Tensor,
    variables: Variables,
    alpha: float,
    mask: Optional[torch.Tensor] = None,
    sum_type: Optional[str] = "all",
) -> torch.Tensor:
    """
    This function computes the negative log likelihood for all features and sums them.

    Args:
        data: Input data, shape (batch_size, input_count).
        decoder_mean: Output from the decoder, shape (batch_size, output_count)
        decoder_logvar: Output from the decoder, shape (batch_size, output_count)
        variables: List of all variables
        alpha: Categorical likelihood coefficient in NLL calculation.
        mask: Mask for input data, shape (batch_size, input_count). 1 is present, 0 is missing. Set to all 1's if None.
        sum_type: How to sum result. None will return the entire array, 'cols' will sum per variable,'all' will sum all
            elements.
    Returns:
        nll: Negative log likelihood summed as per `sum_type`. torch.Tensor of shape (batch_size, num_vars)
        if `sum_type=='all'`, shape (1, num_vars) if `sum_type=='cols'` or a scalar if `sum_type is None`. Note that if
        the data contains categorical variables, then num_vars <= num_features, where num_features is the number of
        features in the input data, since these are encoded using a one-hot encoding which spans multiple columns.
    """
    variables = variables.subset(list(range(0, variables.num_unprocessed_non_aux_cols)))
    assert sum_type in [None, "cols", "all"]
    if data.ndim != 2:  # type: ignore
        raise ValueError("Data should have dims (batch_size, input_count)")
    if decoder_logvar.ndim != 2 or decoder_mean.ndim != 2:  # type: ignore
        raise ValueError("decoder_logvar and decoder_mean should each have dims (batch_size, output_count)")

    batch_size = data.shape[0]
    num_vars = variables.num_unprocessed_cols
    if mask is None:
        mask = torch.ones_like(data)

    # Call unprocessed columns vars, processed columns idxs
    vars_by_type, idxs_by_type = (
        variables.unprocessed_cols_by_type,
        variables.processed_cols_by_type,
    )

    if sum_type is None:
        nlls = torch.zeros(batch_size, num_vars, device=data.device, dtype=data.dtype)
    else:
        nlls = torch.zeros(1, num_vars, device=data.device, dtype=data.dtype)

    def flatten(lists):
        """
        Flatten idxs for continuous and binary vars, since they will be of form [[1], [2], ...]
        """
        return [i for sublist in lists for i in sublist]

    # If returning columnwise/total sum, we sum the NLL for each var. Note if returning the total sum, we don't sum over
    # all elements of each type here, to make it easier to collect everything in a single nlls tensor.
    feature_sum_type = "cols" if sum_type is not None else None
    if "continuous" in vars_by_type:
        continuous_vars, continuous_idxs = (
            vars_by_type["continuous"],
            flatten(idxs_by_type["continuous"]),
        )
        continuous_idxs_nlls = gaussian_negative_log_likelihood(
            data[:, continuous_idxs],
            decoder_mean[:, continuous_idxs],
            decoder_logvar[:, continuous_idxs],
            mask[:, continuous_idxs],
            sum_type=feature_sum_type,
        )
        # Need to account for VAEM's overwrite_processed_dim hack
        # (i.e. continuous variables possible being of dimension>1)
        if all(len(idxs) == 1 for idxs in idxs_by_type["continuous"]):
            # Optimized operation when all continuous variables are of dimension 1
            nlls[:, continuous_vars] = continuous_idxs_nlls
        else:
            # Slower, correct operation if there is continuous variable of dimension > 1
            if len(continuous_idxs_nlls.shape) == 1:
                continuous_idxs_nlls = continuous_idxs_nlls.unsqueeze(dim=0)
            current_idx = 0
            for var, idxs in zip(continuous_vars, idxs_by_type["continuous"]):
                var_idxs = range(current_idx, current_idx + len(idxs))
                nlls[:, var] = continuous_idxs_nlls[:, var_idxs].sum(dim=1)
            current_idx += len(idxs_by_type["continuous"][-1])
    if "binary" in vars_by_type:
        binary_vars, binary_idxs = (
            vars_by_type["binary"],
            flatten(idxs_by_type["binary"]),
        )
        nlls[:, binary_vars] = bernoulli_negative_log_likelihood(
            data[:, binary_idxs],
            decoder_mean[:, binary_idxs],
            mask[:, binary_idxs],
            sum_type=feature_sum_type,
        )
    if "categorical" in vars_by_type:
        categorical_vars, categorical_idxs = (
            vars_by_type["categorical"],
            idxs_by_type["categorical"],
        )
        for var, idxs in zip(categorical_vars, categorical_idxs):
            # Have to compute NLL for each categorical variable separately due to different numbers of categories
            nlls[:, var] = alpha * categorical_negative_log_likelihood(
                data[:, idxs],
                decoder_mean[:, idxs],
                mask[:, idxs],
                sum_type=feature_sum_type,
            )
    # Now sum everything if returning total sum.
    if sum_type == "all":
        nlls = nlls.sum()

    return nlls
