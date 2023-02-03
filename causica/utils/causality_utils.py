import logging
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
import torch
from sklearn import metrics

from ..datasets.dataset import TemporalDataset
from ..datasets.intervention_data import InterventionData
from ..datasets.variables import Variables
from ..models.imodel import (
    IModelForCausalInference,
    IModelForCounterfactuals,
    IModelForInterventions,
    IModelForTimeseries,
)
from ..utils.helper_functions import to_tensors
from ..utils.torch_utils import LinearModel, MultiROFFeaturiser
from .evaluation_dataclasses import AteRMSEMetrics, IteRMSEMetrics, TreatmentDataLogProb
from .nri_utils import (
    convert_temporal_to_static_adjacency_matrix,
    edge_prediction_metrics,
    edge_prediction_metrics_multisample,
    make_temporal_adj_matrix_compatible,
)

logger = logging.getLogger(__name__)


def intervene_graph(
    adj_matrix: torch.Tensor, intervention_idxs: Optional[torch.Tensor], copy_graph: bool = True
) -> torch.Tensor:
    """
    Simulates an intervention by removing all incoming edges for nodes being intervened

    Args:
        adj_matrix: torch.Tensor of shape (input_dim, input_dim) containing  adjacency_matrix
        intervention_idxs: torch.Tensor containing which variables to intervene
        copy_graph: bool whether the operation should be performed in-place or a new matrix greated
    """
    if intervention_idxs is None or len(intervention_idxs) == 0:
        return adj_matrix

    if copy_graph:
        adj_matrix = adj_matrix.clone()

    adj_matrix[:, intervention_idxs] = 0
    return adj_matrix


def intervention_to_tensor(
    intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]],
    intervention_values: Optional[Union[torch.Tensor, np.ndarray]],
    group_mask,
    device,
    is_temporal: bool = False,
) -> Tuple[Optional[torch.Tensor], ...]:
    """
    Maps empty interventions to nan and np.ndarray intervention data to torch tensors.
    Converts indices to a mask using the group_mask. If the intervention format is temporal, set is_temporal to True.
    Args:
        intervention_idxs: np.ndarray or torch.Tensor with shape [num_interventions] (for static data) or [num_interventions, 2] (for temporal data).
        intervention_values: np.ndarray or torch.Tensor with shape [proc_dims] storing the intervention values corresponding to the intervention_idxs.
        group_mask: np.ndarray, a mask of shape (num_groups, num_processed_cols) indicating which column
            corresponds to which group.
        is_temporal: Whether intervention_idxs in temporal 2D format.

    Returns:

    """

    intervention_mask = None

    if intervention_idxs is not None and intervention_values is not None:
        (intervention_idxs,) = to_tensors(intervention_idxs, device=device, dtype=torch.long)
        (intervention_values,) = to_tensors(intervention_values, device=device, dtype=torch.float)

        if intervention_idxs.nelement() == 0:
            intervention_idxs = None

        if intervention_values.nelement() == 0:
            intervention_values = None

        if is_temporal:
            assert intervention_idxs is not None, "For temporal interventions, intervention_idxs must be provided"
            intervention_idxs, intervention_mask, intervention_values = get_mask_and_value_from_temporal_idxs(
                intervention_idxs, intervention_values, group_mask, device
            )
        else:

            intervention_mask = get_mask_from_idxs(intervention_idxs, group_mask, device)

    assert intervention_idxs is None or isinstance(intervention_idxs, torch.Tensor)
    assert intervention_values is None or isinstance(intervention_values, torch.Tensor)
    return intervention_idxs, intervention_mask, intervention_values  # type: ignore


def get_mask_and_value_from_temporal_idxs(
    intervention_idxs: torch.Tensor,
    intervention_values: Optional[torch.Tensor],
    group_mask: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    This is to generate re-ordered intervention_idxs, intervention_mask and intervention_values from original intervention_idxs. This can handle
    arbitrary ordering in original intervention_idxs.
    Args:
        intervention_idxs: torch.Tensor with shape [num_interventions, 2], storing the indices of intervened variable and time step
        intervention_values: torch.Tensor with shape [proc_dim], storing the intervention values corresponding to intervention_idxs. If None, return None as well.
        group_mask: np.ndarray, a mask of shape (num_groups, num_processed_cols) indicating which column
            corresponds to which group.
    Returns:
        intervention_idxs: re-ordered intervention_idxs with shape [num_interventions, 2], corresponding to intervention_mask
        intervention_mask: torch.Tensor, a binary mask with shape [time_length, data_proc_dim]
        intervention_values: torch.Tensor with shape [proc_dim], this is a re-ordered version corresponding to intervention_mask
    """
    # The key issue is how to perform re-ordering according to intervention_mask. We adopt the following approach:
    # (1) Create surrogate_intervention_values with shape [time_length, data_proc_dim] and value nan. (2) iterate through
    # each intervened variables in original intervention_idxs, and assign True to corresponding intervention_mask and corresponding values
    # to surrogate_intervention_values. (3) Then flatten the surrogate_intervention_values and remove all nan values in it, the resulting array will have
    # the order that is consistent with intervention_mask. The re-ordering of intervention_idxs can be done in a similar way, where the surrogate_intervention_idxs
    # (shape [time_length, num_nodes]) contains the indices instead.

    # Initialize the intervention_mask, surrogate_intervention_values and surrogate_intervention_idxs
    num_node, data_proc_dim = group_mask.shape
    time_length = int(torch.max(intervention_idxs[:, -1]).item()) + 1
    intervention_mask = torch.zeros(time_length, data_proc_dim, device=device, dtype=torch.bool)
    surrogate_intervention_value = torch.full((time_length, data_proc_dim), torch.nan, device=device)
    surrogate_intervention_idxs = torch.full((time_length, num_node), torch.nan, device=device)

    # covert group_mask to tensor
    (group_mask_tensor,) = to_tensors(group_mask, device=device, dtype=torch.bool)

    # Iterate through each variables in intervention_idxs
    value_start_idx = 0
    for idx, (variable_idx, variable_time) in enumerate(intervention_idxs):
        # Get binary mask array from group_mask
        row_mask = group_mask_tensor[variable_idx]  # shape [data_proc_dim]
        cur_proc_dim = int(torch.sum(row_mask).item())
        # Get the candidate intervention_mask
        candidate_intervention_mask = torch.zeros(time_length, data_proc_dim, device=device, dtype=torch.bool)
        candidate_intervention_mask[variable_time] = row_mask  # shape [time_length, data_proc_dim]
        # Update intervention_mask
        intervention_mask += candidate_intervention_mask
        if intervention_values is not None:
            # Update surrogate_intervention_values
            surrogate_intervention_value[candidate_intervention_mask] = intervention_values[
                value_start_idx : value_start_idx + cur_proc_dim
            ]
        value_start_idx += cur_proc_dim
        # Update surrogate_intervention_idxs
        surrogate_intervention_idxs[variable_time, variable_idx] = idx

    # Get intervention_mask
    intervention_mask = intervention_mask.bool()  # shape [time_length, data_proc_dim]
    # Generate the re-ordered intervention_idxs, intervention_values
    if intervention_values is not None:
        reordered_intervention_values = surrogate_intervention_value[~torch.isnan(surrogate_intervention_value)]
    else:
        reordered_intervention_values = None

    reordered_intervention_idxs = intervention_idxs[
        surrogate_intervention_idxs[~torch.isnan(surrogate_intervention_idxs)].long()
    ]

    return reordered_intervention_idxs, intervention_mask, reordered_intervention_values


def get_mask_from_idxs(idxs, group_mask, device) -> torch.Tensor:
    """
    Generate mask for observations or samples from indices using group_mask
    """
    mask = torch.zeros(group_mask.shape[0], device=device, dtype=torch.bool)
    mask[idxs] = 1
    (group_mask,) = to_tensors(group_mask, device=device, dtype=torch.bool)
    mask = (mask.unsqueeze(1) * group_mask).sum(0).bool()
    return mask


def get_treatment_data_logprob(
    model: IModelForCausalInference,
    intervention_datasets: List[InterventionData],
    most_likely_graph: bool = False,
) -> TreatmentDataLogProb:
    """
    Computes the log-probability of test-points sampled from intervened distributions.
    Args:
        model: IModelForInterventions with which we can evaluate the log-probability of points while applying interventions to the generative model
        intervention_datasets: List[InterventionData] containing intervetions and samples from the ground truth data generating process when the intervention is applied
        most_likely_graph: whether to use the most likely causal graph (True) or to sample graphs (False)
    Returns:
        Summary statistics about the log probabilities from the intervened distributions
    """
    all_log_probs = []
    assert isinstance(model, IModelForInterventions)
    for intervention_data in intervention_datasets:

        assert intervention_data.intervention_values is not None
        intervention_log_probs = model.log_prob(
            X=intervention_data.test_data.astype(float),
            most_likely_graph=most_likely_graph,
            intervention_idxs=intervention_data.intervention_idxs,
            intervention_values=intervention_data.intervention_values,
        )
        # Evaluate log-prob per dimension
        assert intervention_data.intervention_idxs is not None
        intervention_log_probs = intervention_log_probs / (
            intervention_data.test_data.shape[1] - intervention_data.intervention_idxs.shape[0]
        )

        all_log_probs.append(intervention_log_probs)

    per_intervention_log_probs_mean = [log_probs.mean(axis=0) for log_probs in all_log_probs]
    per_intervention_log_probs_std = [log_probs.std(axis=0) for log_probs in all_log_probs]

    if len(all_log_probs) > 0:
        all_log_probs_arr = np.concatenate(all_log_probs, axis=0)
    else:
        all_log_probs_arr = np.array([np.nan])

    return TreatmentDataLogProb(
        all_mean=all_log_probs_arr.mean(axis=0),
        all_std=all_log_probs_arr.std(axis=0),
        per_intervention_mean=per_intervention_log_probs_mean,
        per_intervention_std=per_intervention_log_probs_std,
    )


def get_ate_rms(
    model: IModelForInterventions,
    test_samples: np.ndarray,
    intervention_datasets: List[InterventionData],
    variables: Variables,
    most_likely_graph: bool = False,
    processed: bool = True,
) -> Tuple[AteRMSEMetrics, AteRMSEMetrics]:
    """
    Computes the rmse between the ground truth ate and the ate predicted by our model across all available interventions
        for both normalised and unnormalised data.
    Args:
        model: IModelForInterventions from which we can sample points while applying interventions
        test_samples: np.ndarray of shape (Nsamples, observation_dimension) containing samples from the non-intervened distribution p(y)
        intervention_datasets: List[InterventionData] containing intervetions and samples from the ground truth data generating process when the intervention is applied
        variables: Instance of Variables containing metadata used for normalisation
        most_likely_graph: whether to use the most likely causal graph (True) or to sample graphs (False)
        processed: whether the data has been processed
    Returns:
        Root mean square errors for both normalised and unnormalised data
    """

    group_rmses = []
    norm_group_rmses = []

    for intervention_data in intervention_datasets:

        if intervention_data.reference_data is not None:
            reference_data = intervention_data.reference_data
        else:
            reference_data = test_samples

        # conditions are applied to the test data when it is generated. As a result computing ATE on this data returns the CATE.
        ate = get_ate_from_samples(
            intervention_data.test_data, reference_data, variables, normalise=False, processed=processed
        )
        norm_ate = get_ate_from_samples(
            intervention_data.test_data, reference_data, variables, normalise=True, processed=processed
        )

        # Filter effect groups
        if intervention_data.effect_idxs is not None:
            [ate, norm_ate], filtered_variables = filter_effect_groups(
                [ate, norm_ate], variables, intervention_data.effect_idxs, processed
            )

        else:
            filtered_variables = variables

        # Check for conditioning
        if intervention_data.conditioning_idxs is not None:
            if most_likely_graph:
                Ngraphs = 1
                Nsamples_per_graph = 50000
            else:
                Ngraphs = 10
                Nsamples_per_graph = 5000
        else:
            if most_likely_graph:
                Ngraphs = 1
                Nsamples_per_graph = 20000
            else:
                Ngraphs = 10000
                Nsamples_per_graph = 2

        assert intervention_data.intervention_values is not None

        if isinstance(model, IModelForTimeseries):
            model_ate, model_norm_ate = model.cate(
                intervention_idxs=np.array(intervention_data.intervention_idxs),
                intervention_values=intervention_data.intervention_values,
                reference_values=intervention_data.intervention_reference,
                effect_idxs=intervention_data.effect_idxs,
                conditioning_idxs=None,
                conditioning_values=None,
                most_likely_graph=most_likely_graph,
                Nsamples_per_graph=Nsamples_per_graph,
                Ngraphs=Ngraphs,
                conditioning_history=intervention_data.conditioning_values,
            )
        else:
            model_ate, model_norm_ate = model.cate(
                intervention_idxs=np.array(intervention_data.intervention_idxs),
                intervention_values=intervention_data.intervention_values,
                reference_values=intervention_data.intervention_reference,
                effect_idxs=intervention_data.effect_idxs,
                conditioning_idxs=intervention_data.conditioning_idxs,
                conditioning_values=intervention_data.conditioning_values,
                most_likely_graph=most_likely_graph,
                Nsamples_per_graph=Nsamples_per_graph,
                Ngraphs=Ngraphs,
            )

        model_ate = np.atleast_1d(model_ate)
        model_norm_ate = np.atleast_1d(model_norm_ate)
        group_rmses.append(calculate_per_group_rmse(model_ate, ate, filtered_variables))
        norm_group_rmses.append(calculate_per_group_rmse(model_norm_ate, norm_ate, filtered_variables))

    return AteRMSEMetrics(np.stack(group_rmses, axis=0)), AteRMSEMetrics(np.stack(norm_group_rmses, axis=0))


def get_ate_from_samples(
    intervened_samples: np.ndarray,
    baseline_samples: np.ndarray,
    variables: Variables,
    normalise: bool = False,
    processed: bool = True,
):
    """
    Computes ATE E[y | do(x)=a] - E[y] from samples of y from p(y | do(x)=a) and p(y)

    Args:
        intervened_samples: np.ndarray of shape (Nsamples, observation_dimension) containing samples from the intervened distribution p(y | do(x)=a)
        baseline_samples: np.ndarray of shape (Nsamples, observation_dimension) containing samples from the non-intervened distribution p(y)
        variables: Instance of Variables containing metada used for normalisation
        normalise: boolean indicating whether to normalise samples by their maximum and minimum values
        processed: whether the data has been processed (which affects the column numbering)
    """
    if normalise:
        assert variables is not None, "must provide an associated Variables instance to perform normalisation"
        intervened_samples, baseline_samples = normalise_data(
            [intervened_samples, baseline_samples], variables, processed
        )

    intervened_mean = intervened_samples.mean(axis=0)
    baseline_mean = baseline_samples.mean(axis=0)

    return intervened_mean - baseline_mean


def get_cate_from_samples(
    intervened_samples: torch.Tensor,
    baseline_samples: torch.Tensor,
    conditioning_mask: torch.Tensor,
    conditioning_values: torch.Tensor,
    effect_mask: torch.Tensor,
    variables: Optional[Variables] = None,
    normalise: bool = False,
    rff_lengthscale: Union[int, float, List[float], Tuple[float, ...]] = (0.1, 1),
    rff_n_features: int = 3000,
):
    """
    Estimate CATE using a functional approach: We fit a function that takes as input the conditioning variables
     and as output the outcome variables using intervened_samples as training points. We do the same while using baseline_samples
     as training data. We estimate CATE as the difference between the functions' outputs when the input is set to conditioning_values.
     As functions we use linear models on a random fourier feature basis. If intervened_samples and baseline_samples are provided for multiple graphs
     the CATE estimate is averaged across graphs.

    Args:
        intervened_samples: tensor of shape (Ngraphs, Nsamples, Nvariables) sampled from intervened (non-conditional) distribution
        baseline_samples: tensor of shape (Ngraphs, Nsamples, Nvariables) sampled from a reference distribution. Note that this could mean a reference intervention has been applied.
        conditioning_mask: boolean tensor which indicates which variables we want to condition on
        conditioning_values: tensor containing values of variables we want to condition on
        effect_mask: boolean tensor which indicates which outcome variables for which we want to estimate CATE
        variables: Instance of Variables containing metada used for normalisation
        normalise: boolean indicating whether to normalise samples by their maximum and minimum values
        rff_lengthscale: either a positive float/int indicating the lengthscale of the RBF kernel or a list/tuple
         containing the lower and upper limits of a uniform distribution over the lengthscale. The latter option is prefereable when there is no prior knowledge about functional form.
        rff_n_features: Number of random features with which to approximate the RBF kernel. Larger numbers result in lower variance but are more computationally demanding.
    Returns:
        CATE_estimates: tensor of shape (len(effect_idxs)) containing our estimates of CATE for outcome variables
    """

    # TODO: we are assuming the conditioning variable is d-connected to the target but we should probably use the networkx dseparation method to check this in the future
    if normalise:
        assert variables is not None, "must provide an associated Variables instance to perform normalisation"
        intervened_samples_np, baseline_samples_np = normalise_data(
            [intervened_samples.cpu().numpy(), baseline_samples.cpu().numpy()], variables, processed=True
        )

        # Convert back to tensors
        intervened_samples = torch.tensor(
            intervened_samples_np, device=intervened_samples.device, dtype=intervened_samples.dtype
        )
        baseline_samples = torch.tensor(
            baseline_samples_np, device=baseline_samples.device, dtype=baseline_samples.dtype
        )

    assert effect_mask.sum() == 1.0, "Only 1d outcomes are supported"

    featuriser = MultiROFFeaturiser(rff_n_features=rff_n_features, lengthscale=rff_lengthscale)
    featuriser.fit(X=intervened_samples.new_ones((1, int(conditioning_mask.sum()))))

    CATE_estimates = []
    for graph_idx in range(intervened_samples.shape[0]):
        intervened_train_inputs = intervened_samples[graph_idx, :, conditioning_mask]
        reference_train_inputs = baseline_samples[graph_idx, :, conditioning_mask]

        featurised_intervened_train_inputs = featuriser.transform(intervened_train_inputs)
        featurised_reference_train_inputs = featuriser.transform(reference_train_inputs)
        featurised_conditioning_values = featuriser.transform(torch.atleast_1d(conditioning_values))

        intervened_train_targets = intervened_samples[graph_idx, :, effect_mask].reshape(intervened_samples.shape[1])
        reference_train_targets = baseline_samples[graph_idx, :, effect_mask].reshape(intervened_samples.shape[1])

        intervened_predictive_model = LinearModel()
        intervened_predictive_model.fit(features=featurised_intervened_train_inputs, targets=intervened_train_targets)

        reference_predictive_model = LinearModel()
        reference_predictive_model.fit(features=featurised_reference_train_inputs, targets=reference_train_targets)

        CATE_estimates.append(
            intervened_predictive_model.predict(features=featurised_conditioning_values)[0]
            - reference_predictive_model.predict(features=featurised_conditioning_values)[0]
        )

    return torch.stack(CATE_estimates, dim=0).mean(dim=0)


def get_ite_from_samples(
    intervention_samples: np.ndarray,
    reference_samples: np.ndarray,
    variables: Optional[Variables] = None,
    normalise: bool = False,
    processed: bool = True,
):
    """
    Calculates individual treatment effect (ITE) between two sets of samples each
    with shape (no. of samples, no. of variables).

    Args:
        intervention_samples (ndarray): Samples from intervened graph with shape (no. of samples, no. of dimenions).
        reference_samples (ndarray): Reference samples from intervened graph with shape (no. of samples, no. of dimenions).
        variables (Variables): A `Variables` instance relating to passed samples.
        normalised (bool): Flag indicating whether the data should be normalised (using `variables`) prior to
            calculating ITE.
        processed (bool): Flag indicating whether the passed samples have been processed.

    Returns: ITE with shape (no. of samples, no. of variables)
    """
    if normalise:
        assert variables is not None, "must provide an associated Variables instance to perform normalisation"
        intervention_samples, reference_samples = normalise_data(
            [intervention_samples, reference_samples], variables, processed
        )

    assert (
        intervention_samples.shape == reference_samples.shape
    ), "Intervention and reference samples must be the shape for ITE calculation"
    return intervention_samples - reference_samples


def calculate_rmse(a: np.ndarray, b: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Calculates the root mean squared error (RMSE) between arrays `a` and `b`.

    Args:
        a (ndarray): Array used for error calculation
        b (ndarray): Array used for error calculation
        axis (int): Axis upon which to calculate mean

    Returns: (ndarray) RMSE value taken along axis `axis`.
    """
    return np.sqrt(np.mean(np.square(np.subtract(a, b)), axis=axis))


def normalise_data(arrs: List[np.ndarray], variables: Variables, processed: bool) -> List[np.ndarray]:
    """
    Normalises all arrays in `arrs` to [0, 1] given variable maximums (upper) and minimums (lower)
    in `variables`. Categorical data is excluded from normalization.

    Args:
        arrs (List[ndarray]): A list of ndarrays to normalise
        variables (Variables): A Variables instance containing metadata about arrays in `arrs`
        processed (bool): Whether the data in `arrs` has been processed

    Returns:
        (list(ndarray)) A list of normalised ndarrays corresponding with `arrs`.
    """

    if processed:
        n_cols = variables.num_processed_cols
        col_groups = variables.processed_cols
    else:
        n_cols = variables.num_unprocessed_cols
        col_groups = variables.unprocessed_cols

    assert all(
        n_cols == arr.shape[-1] for arr in arrs
    ), f"Expected {n_cols} columns for the passed {'' if processed else 'non-'}processed array"

    # if lower/uppers aren't updated, performs (arr - 0)/(1 - 0), i.e. doesn't normalize
    lowers = np.zeros(n_cols)
    uppers = np.ones(n_cols)

    for cols_idx, variable in zip(col_groups, variables):
        if variable.type_ == "continuous":
            lowers[cols_idx] = variable.lower
            uppers[cols_idx] = variable.upper

    return [np.divide(np.subtract(arr, lowers), np.subtract(uppers, lowers)) for arr in arrs]


def calculate_per_group_rmse(a: np.ndarray, b: np.ndarray, variables: Variables) -> np.ndarray:
    """
    Calculates RMSE group-wise between two ndarrays (`a` and `b`) for all samples.
    Arrays 'a' and 'b' have expected shape (no. of rows, no. of variables) or (no. of variables).

    Args:
        a (ndarray): Array of shape (no. of rows, no. of variables)
        b (ndarray): Array of shape (no. of rows, no. of variables)
        variables (Variables): A Variables object indicating groups

    Returns:
        (ndarrray) RMSE calculated over each group for each sample in `a`/`b`
    """
    rmse_array = np.zeros((a.shape[0], variables.num_groups) if len(a.shape) == 2 else (variables.num_groups))
    for return_array_idx, group_idxs in enumerate(variables.group_idxs):
        # calculate RMSE columnwise for all samples
        rmse_array[..., return_array_idx] = calculate_rmse(a[..., group_idxs], b[..., group_idxs], axis=-1)
    return rmse_array


def filter_effect_groups(
    arrs: List[np.ndarray], variables: Variables, effect_group_idxs: np.ndarray, processed: bool
) -> Tuple[List[np.ndarray], Variables]:
    """
    Returns the groups (nodes) associated with effect variables. If `proccessed` is True, assume
    that arrs has been processed and handle expanded columns appropriately.

    Args:
        arrs (List[ndarray]): A list of ndarrays to be filtered
        variables (Variables): A Variables instance containing metadata
        effect_group_idxs (np.ndarray): An array containing idxs of effect variables
        processed (bool): Whether to treat data in `arrs` as having been processed

    Returns: A list of ndarrays corresponding to `arrs` with columns relating to effect variables,
        and a new Variables instance relating to effect variables
    """
    if effect_group_idxs.ndim == 2:
        # temporal
        temporal_idxs = effect_group_idxs[:, 1]

        arrs = [a[..., temporal_idxs, :] for a in arrs]

        effect_group_idxs = effect_group_idxs[:, 0]

    effect_variable_idxs = [j for i in effect_group_idxs for j in variables.group_idxs[i]]
    if processed:
        # Get effect idxs according to processed data
        processed_effect_idxs = []
        for i in effect_variable_idxs:
            processed_effect_idxs.extend(variables.processed_cols[i])
    else:
        processed_effect_idxs = effect_variable_idxs

    return [a[..., processed_effect_idxs] for a in arrs], variables.subset(effect_variable_idxs)


def get_ite_evaluation_results(
    model: IModelForInterventions,
    counterfactual_datasets: List[InterventionData],
    variables: Variables,
    processed: bool,
    most_likely_graph: bool = False,
    Ngraphs: int = 100,
) -> Tuple[IteRMSEMetrics, IteRMSEMetrics]:
    """
    Calculates ITE evaluation metrics. Only evaluates target variables indicated in `variables`,
    if no target variables are indicate then evaluates all variables.

    Args:
        model (IModelForinterventions): Trained DECI model
        counterfactual_datasets (list[InterventionData]): a list of counterfactual datasets
            used to calculate metrics.
        variables (Variables): Variables object indicating variable group membership
        normalise (bool): Whether the data should be normalised prior to calculating RMSE
        processed (bool): Whether the data in `counterfactual_datasets` has been processed
        most_likely_graph (bool): Flag indicating whether to use most likely graph.
            If false, model-generated counterfactual samples are averaged over `Ngraph` graphs.
        Ngraphs (int): Number of graphs sampled when generating counterfactual samples. Unused if
            `most_likely_graph` is true.

    Returns:
            IteEvaluationResults object containing ITE evaluation metrics.
    """

    group_rmses = []
    norm_group_rmses = []
    for counterfactual_int_data in counterfactual_datasets:
        baseline_samples = counterfactual_int_data.conditioning_values
        reference_samples = counterfactual_int_data.reference_data
        intervention_samples = counterfactual_int_data.test_data
        assert intervention_samples is not None
        assert reference_samples is not None

        # get sample (ground truth) ite
        sample_ite = get_ite_from_samples(
            intervention_samples=intervention_samples,
            reference_samples=reference_samples,
            variables=variables,
            normalise=False,
            processed=processed,
        )

        sample_norm_ite = get_ite_from_samples(
            intervention_samples=intervention_samples,
            reference_samples=reference_samples,
            variables=variables,
            normalise=True,
            processed=processed,
        )
        assert isinstance(model, IModelForCounterfactuals)
        assert counterfactual_int_data.intervention_values is not None
        assert baseline_samples is not None
        # get model (predicted) ite
        model_ite, model_norm_ite = model.ite(
            X=baseline_samples,
            intervention_idxs=counterfactual_int_data.intervention_idxs,
            intervention_values=counterfactual_int_data.intervention_values,
            reference_values=counterfactual_int_data.intervention_reference,
            most_likely_graph=most_likely_graph,
            Ngraphs=Ngraphs,
        )

        # if there are defined target variables, only use these for evaluation
        if counterfactual_int_data.effect_idxs is not None:
            arrs = [sample_ite, model_ite, sample_norm_ite, model_norm_ite]
            filtered_arrs, filtered_variables = filter_effect_groups(
                arrs, variables, counterfactual_int_data.effect_idxs, processed
            )
            [sample_ite, model_ite, sample_norm_ite, model_norm_ite] = filtered_arrs
        else:
            filtered_variables = variables

        # calculate ite rmse per group for current intervention
        # (no. of samples, no. of input variables) -> (no. of samples, no. of groups)
        group_rmses.append(calculate_per_group_rmse(sample_ite, model_ite, filtered_variables))
        norm_group_rmses.append(calculate_per_group_rmse(sample_norm_ite, model_norm_ite, filtered_variables))

    return IteRMSEMetrics(np.stack(group_rmses, axis=0)), IteRMSEMetrics(np.stack(norm_group_rmses, axis=0))


def calculate_regret(variables: Variables, X: torch.Tensor, target_idx: int, max_values: torch.Tensor) -> torch.Tensor:
    """Computes the regret function, given an array of maximum values.

    The regret is defined as

        regret(X) = max_values(X) - observed_outcome

    where `max_values(X)` is the maximum attainable value at `X`, which should be provided.
    This can be computed either with `posterior_expected_optimal_policy`, or by a user-defined method.

    Args:
        X: tensor of shape (num_samples, processed_dim_all) containing the contexts and observed outcomes
        target_idx: index of the target (outcome) variable. Should be 0 <= target_idx < num_nodes.
        max_values: tensor of shape (num_samples) containing the maximum value for each context. The ordering of rows
            should match with `X`.

    Returns:
        regret_values: tensor of shape (num_samples) containing the regret.
    """
    target_mask = get_mask_from_idxs([target_idx], group_mask=variables.group_mask, device=X.device)
    observed_values = X[..., target_mask].squeeze(-1)
    return max_values - observed_values


def dag_pen_np(X):
    assert X.shape[0] == X.shape[1]
    X = torch.from_numpy(X)
    return (torch.trace(torch.matrix_exp(X)) - X.shape[0]).item()


def int2binlist(i: int, n_bits: int):
    """
    Convert integer to list of ints with values in {0, 1}
    """
    assert i < 2**n_bits
    str_list = list(np.binary_repr(i, n_bits))
    return [int(i) for i in str_list]


def approximate_maximal_acyclic_subgraph(adj_matrix: np.ndarray, n_samples: int = 10) -> np.ndarray:
    """
    Compute an (approximate) maximal acyclic subgraph of a directed non-dag but removing at most 1/2 of the edges
    See Vazirani, Vijay V. Approximation algorithms. Vol. 1. Berlin: springer, 2001, Page 7;
    Also Hassin, Refael, and Shlomi Rubinstein. "Approximations for the maximum acyclic subgraph problem."
    Information processing letters 51.3 (1994): 133-140.
    Args:
        adj_matrix: adjacency matrix of a directed graph (may contain cycles)
        n_samples: number of the random permutations generated. Default is 10.
    Returns:
        an adjacency matrix of the acyclic subgraph
    """
    # assign each node with a order
    adj_dag = np.zeros_like(adj_matrix)
    for _ in range(n_samples):
        random_order = np.expand_dims(np.random.permutation(adj_matrix.shape[0]), 0)
        # subgraph with only forward edges defined by the assigned order
        adj_forward = ((random_order.T > random_order).astype(int)) * adj_matrix
        # subgraph with only backward edges defined by the assigned order
        adj_backward = ((random_order.T < random_order).astype(int)) * adj_matrix
        # return the subgraph with the least deleted edges
        adj_dag_n = adj_forward if adj_backward.sum() < adj_forward.sum() else adj_backward
        if adj_dag_n.sum() > adj_dag.sum():
            adj_dag = adj_dag_n
    return adj_dag


def cpdag2dags(cp_mat: np.ndarray, samples: Optional[int] = None) -> np.ndarray:
    """
    Compute all possible DAGs contained within a Markov equivalence class, given by a CPDAG
    Args:
        cp_mat: adjacency matrix containing both forward and backward edges for edges for which directionality is undetermined
    Returns:
        3 dimensional tensor, where the first indexes all the possible DAGs
    """
    assert len(cp_mat.shape) == 2 and cp_mat.shape[0] == cp_mat.shape[1]

    # matrix composed of just undetermined edges
    cycle_mat = (cp_mat == cp_mat.T) * cp_mat
    # return original matrix if there are no length-1 cycles
    if cycle_mat.sum() == 0:
        if dag_pen_np(cp_mat) != 0.0:
            cp_mat = approximate_maximal_acyclic_subgraph(cp_mat)
        return cp_mat[None, :, :]

    # matrix of determined edges
    cp_determined_subgraph = cp_mat - cycle_mat

    # prune cycles if the matrix of determined edges is not a dag
    if dag_pen_np(cp_determined_subgraph.copy()) != 0.0:
        cp_determined_subgraph = approximate_maximal_acyclic_subgraph(cp_determined_subgraph, 1000)

    # number of parent nodes for each node under the well determined matrix
    N_in_nodes = cp_determined_subgraph.sum(axis=0)

    # lower triangular version of cycles edges: only keep cycles in one direction.
    cycles_tril = np.tril(cycle_mat, k=-1)

    # indices of potential new edges
    undetermined_idx_mat = np.array(np.nonzero(cycles_tril)).T  # (N_undedetermined, 2)

    # number of undetermined edges
    N_undetermined = int(cycles_tril.sum())

    # choose random order for mask iteration
    max_dags = 2**N_undetermined

    if max_dags > 10000:
        logger.warning("The number of possible dags are too large (>10000), limit to 10000")
        max_dags = 10000

    if samples is None:
        samples = max_dags
    mask_indices = list(np.random.permutation(np.arange(max_dags)))

    # iterate over list of all potential combinations of new edges. 0 represents keeping edge from upper triangular and 1 from lower triangular
    dag_list: list = []
    while mask_indices and len(dag_list) < samples:

        mask_index = mask_indices.pop()
        mask = np.array(int2binlist(mask_index, N_undetermined))

        # extract list of indices which our new edges are pointing into
        incoming_edges = np.take_along_axis(undetermined_idx_mat, mask[:, None], axis=1).squeeze()

        # check if multiple edges are pointing at same node
        _, unique_counts = np.unique(incoming_edges, return_index=False, return_inverse=False, return_counts=True)

        # check if new colider has been created by checkig if multiple edges point at same node or if new edge points at existing child node
        new_colider = np.any(unique_counts > 1) or np.any(N_in_nodes[incoming_edges] > 0)

        if not new_colider:
            # get indices of new edges by sampling from lower triangular mat and upper triangular according to indices
            edge_selection = undetermined_idx_mat.copy()
            edge_selection[mask == 0, :] = np.fliplr(edge_selection[mask == 0, :])

            # add new edges to matrix and add to dag list
            new_dag = cp_determined_subgraph.copy()
            new_dag[(edge_selection[:, 0], edge_selection[:, 1])] = 1

            # Check for high order cycles
            if dag_pen_np(new_dag.copy()) == 0.0:
                dag_list.append(new_dag)
    # When all combinations of new edges create cycles, we will only keep determined ones
    if len(dag_list) == 0:
        dag_list.append(cp_determined_subgraph)

    return np.stack(dag_list, axis=0)


def admg2dag(directed_adj: torch.Tensor, bidirected_adj: torch.Tensor) -> torch.Tensor:
    """Converts the ADNG specified by directed_adj and bidirected_adj into a DAG.

    Args:
        directed_adj: Directed adjacency matrix over the observed variables.
        bidirected_adj: Bidirected adjacency matrix over the observed variables.

    Returns:
        The DAG represented by directed_adj and bidirected_adj.
    """
    assert directed_adj.shape == bidirected_adj.shape
    assert len(directed_adj.shape) == 2
    assert directed_adj.shape[0] == directed_adj.shape[1]

    num_observed = directed_adj.shape[-1]
    num_latent = num_observed * (num_observed - 1) // 2
    num_nodes = num_latent + num_observed
    adj_tensor = torch.zeros((num_nodes, num_nodes), device=directed_adj.device)

    # Add directed edges
    adj_tensor[:num_observed, :num_observed] = directed_adj

    # Add bidirected edges.
    for idx1 in range(1, num_observed):
        i = idx1 * (idx1 - 1) // 2
        for idx2 in range(idx1):
            j = num_observed + i + idx2
            adj_tensor[j, idx1] = bidirected_adj[idx1, idx2]
            adj_tensor[j, idx2] = bidirected_adj[idx1, idx2]

    return adj_tensor


def dag2admg(adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Converts a DAG over n observed variables and n * (n - 1) // 2 latent variables into a directed and bidirected
    adjacency matrix over the observed variables.

    Args:
        adj: The adjacency matrix.

    Returns:
        (directed_adj, bidirected_adj) representing the ADMG parameterisation of the DAG.
    """
    assert len(adj.shape) == 2
    assert adj.shape[0] == adj.shape[1]
    d = adj.shape[0]
    n_float = -0.5 + np.sqrt(0.25 + 2 * d)
    assert np.isclose(n_float, round(n_float)), "n is not an integer."

    n = round(n_float)

    # Check latent variables are confounders.
    assert np.all(adj[:, n:].detach().cpu().numpy() == 0), "Latent variables are not confounders."

    directed_adj = adj[:n, :n]
    bidirected_adj = torch.zeros(n, n, device=adj.device)

    for idx1 in range(1, n):
        for idx2 in range(idx1):
            i = n + idx1 * (idx1 - 1) // 2 + idx2
            bidirected_adj[idx1, idx2] = bidirected_adj[idx2, idx1] = adj[i, idx1]

    return directed_adj, bidirected_adj


def pag_to_admg_possibilities(adjacency_pag: np.ndarray, idx1: int, idx2: int):
    """Computes all the directed / bidirected edge possibilities for a given entry in a PAG."""
    # Check idx1 0-0 idx2.
    if adjacency_pag[idx1, idx2] == 2 and adjacency_pag[idx2, idx1] == 2:
        return [((1, 0), 0), ((0, 1), 0), ((0, 0), 1), ((1, 0), 1), ((0, 1), 1)]

    # Check idx1 0-> idx2.
    elif adjacency_pag[idx1, idx2] == 1 and adjacency_pag[idx2, idx1] == 2:
        return [((1, 0), 0), ((0, 0), 1), ((1, 0), 1)]

    # Check idx1 <-0 idx2.
    elif adjacency_pag[idx1, idx2] == 2 and adjacency_pag[idx2, idx1] == 1:
        return [((0, 1), 0), ((0, 0), 1), ((0, 1), 1)]

    # Check idx1 <-> idx2.
    elif adjacency_pag[idx1, idx2] == 1 and adjacency_pag[idx2, idx1] == 1:
        return [((0, 0), 1)]

    # Check idx1 -> idx2.
    elif adjacency_pag[idx1, idx2] == 1:
        return [((1, 0), 0)]

    # Check idx1 <- idx2.
    elif adjacency_pag[idx1, idx2] == 0 and adjacency_pag[idx2, idx1] == 1:
        return [((0, 1), 0)]

    # Check no causal link.
    else:
        return [((0, 0), 0)]


def build_admg_from_edge_info(
    edge_info: List[Tuple[Tuple[int, int], int]], nodes_list: List[Tuple[int, int]], num_nodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Constructs an ADMG from edge information."""
    directed_adj = np.zeros((num_nodes, num_nodes))
    bidirected_adj = np.zeros((num_nodes, num_nodes))

    for nodes, edge in zip(nodes_list, edge_info):
        idx1, idx2 = nodes[0], nodes[1]
        directed_adj[idx1, idx2] = edge[0][0]
        directed_adj[idx2, idx1] = edge[0][1]
        bidirected_adj[idx1, idx2] = bidirected_adj[idx2, idx1] = edge[1]

    return directed_adj, bidirected_adj


def pag2admgs(adjacency_pag: np.ndarray, samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute all possible ADMG graphs contained within a Markov equivalence class, given by a PAG.

    Args:
        adjacency_pag: Adjacency matrix containing both directed and bidirected edges.
        samples: If specified, selects a random subset of possible admgs of size samples.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Two 3 dimensional arrays, the first being the directed graphs and the latter
        being the corresponding bidirected graphs.
    """
    assert len(adjacency_pag.shape) == 2 and adjacency_pag.shape[0] == adjacency_pag.shape[1]

    # List of possible entries in the directed and bidirected graphs for each pair of nodes.
    possibilities_list = []
    idxs_list = []

    for idx1 in range(len(adjacency_pag)):
        for idx2 in range(idx1 + 1, len(adjacency_pag)):
            possibilities_list.append(pag_to_admg_possibilities(adjacency_pag, idx1, idx2))
            idxs_list.append((idx1, idx2))

    admg_edge_list = list(product(*possibilities_list))
    max_admgs = len(admg_edge_list)
    if samples is None:
        samples = max_admgs

    samples = min(samples, max_admgs)

    admg_idxs = list(np.random.permutation(np.arange(max_admgs)))

    # Iterate over list of all potential combinations of new edges.
    directed_adj_list: list = []
    bidirected_adj_list: list = []
    admg_idxs = admg_idxs[:samples]
    for admg_idx in admg_idxs:
        # Get ADMG edge definition.
        admg_edges = admg_edge_list[admg_idx]

        # Build ADMG from edge definitions.
        new_directed_adj, new_bidirected_adj = build_admg_from_edge_info(admg_edges, idxs_list, adjacency_pag.shape[0])

        # Check for higher order cycles.
        if dag_pen_np(new_directed_adj.copy()) == 0.0:
            directed_adj_list.append(new_directed_adj)
            bidirected_adj_list.append(new_bidirected_adj)

    if not directed_adj_list:
        directed_adj_list = [np.zeros_like(adjacency_pag)]
    if not bidirected_adj_list:
        bidirected_adj_list = [np.zeros_like(adjacency_pag)]

    return np.stack(directed_adj_list, axis=0), np.stack(bidirected_adj_list, axis=0)


def process_adjacency_mats(adj_mats: np.ndarray, num_nodes: int):
    """
    This processes the adjacency matrix in the format [num, variable, variable]. It will remove the duplicates and non DAG adjacency matrix.
    Args:
        adj_mats (np.ndarry): A group of adjacency matrix
        num_nodes (int): The number of variables (dimensions of the adjacency matrix)

    Returns:
        A list of adjacency matrix without duplicates and non DAG
        A np.ndarray storing the weights of each adjacency matrix.
    """

    # This method will get rid of the non DAG and duplicated ones. It also returns a proper weight for each of the adjacency matrix
    if len(adj_mats.shape) == 2:
        # Single adjacency matrix
        assert (np.trace(scipy.linalg.expm(adj_mats)) - num_nodes) == 0, "Generate non DAG graph"
        return adj_mats, np.ones(1)
    else:
        # Multiple adjacency matrix samples
        # Remove non DAG adjacency matrix
        adj_mats = np.array(
            [adj_mat for adj_mat in adj_mats if (np.trace(scipy.linalg.expm(adj_mat)) - num_nodes) == 0]
        )
        assert np.any(adj_mats), "Generate non DAG graph"
        # Remove duplicated adjacency and aggregate the weights
        adj_mats_unique, dup_counts = np.unique(adj_mats, axis=0, return_counts=True)
        # Normalize the weights
        adj_weights = dup_counts / np.sum(dup_counts)
        return adj_mats_unique, adj_weights


def get_ipw_estimated_ate(
    interventional_X: Union[torch.Tensor, np.ndarray],
    treatment_probability: Union[torch.Tensor, np.ndarray],
    treatment_mask: Union[torch.Tensor, np.ndarray],
    outcome_idxs: Union[torch.Tensor, np.ndarray],
) -> np.ndarray:
    """Calculate the inverse probability weighted estimated ATE of the interventional data gather from
    either real-world experiments, or simulators.

    This is given by equation (6) of the write-up, which is defined as
    hat{ATE}: = sum_i 1/p_i * T_i Y_i/N_data  -  sum_i 1/(1-p_i) * (1-T_i) Y_i/N_data.

    Args:
        interventional_X: the tensor of shape (num_observations, input_dim) containing the
        interventional/counterfactual observations that are gathered from the real environment/simulator,
        after applying assigned treatments.
        treatment_probability: the tensor of  shape (num_obserbations,) containing the probability (induced by
        test policy) that each subject (row) of interventional X is assigned with treatment combinations at
        intervention values.
            hence 1-treatment_probability will be the probability that each subject (row) of interventional X is
            assigned with treatment combinations at reference values.
            Note that the actual treatment for each subject will be sampled and consolidated in this function.
        treatment_mask:  tensor of binary elements of shape (num_observations,) cotaining the simulated
           treatment assighment for each observation. treatment_mask_i = 1 indicates that subjet i is assigned
           with treatment of value intervention_values; otherwise reference_values is assigned.
        outcome_idxs: torch Tensor of shape (num_targets) containing indices of variables that specifies the outcome Y

    Returns:
        ndarray of shape (num_targets,) containing the ipw estimated value of ATE.
    """
    raise NotImplementedError


def get_real_world_testing_assignment(
    observational_X: Union[torch.Tensor, np.ndarray],
    intervention_idxs: Union[torch.Tensor, np.ndarray],
    intervention_values: Union[torch.Tensor, np.ndarray],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the treatment assigment probability for each subjects.
    This is defined as
    p(T_1=intervention_values_1, T_2=interventional_values_2, ..., |X).

    This can be implemented in two ways:
     1), A simple probability distribution given by certain rules. e.g., random A/B testing
     2), More complicated ones that implicitly depend on another deci model. In this case it should take as input a
     model instance.

    Args:
        observational_X: the tensor of shape (num_observations, input_dim) containing the observational data
        gathered from the real environment/simulator. Note that there aren't any interventions applied yet.
        intervention_idxs: torch.Tensor of shape (num_interventions) containing indices of variables that have
        been intervened.
        intervention_values: torch.Tensor of shape (num_interventons) optional array containing values for
        variables that have been intervened.
    Returns:
        Tuple (treatment_probability, treatment_mask).
           treatment_probability is a tensor of shape (num_observations,) containing the treatment_probability
           for each observation.
           treatment_mask is a tensor of binary elements of shape (num_observations,) cotaining the simulated
           treatment assighment for each observation. treatment_mask_i = 1 indicates that subjet i is assigned
           with treatment of value intervention_values; otherwise reference_values is assigned.

    """
    raise NotImplementedError


def eval_test_quality_by_ate_error(
    test_environment: IModelForCounterfactuals,
    observational_X: Union[torch.Tensor, np.ndarray],
    treatment_probability: Union[torch.Tensor, np.ndarray],
    treatment_mask: Union[torch.Tensor, np.ndarray],
    outcome_idxs: Union[torch.Tensor, np.ndarray],
    intervention_idxs: Union[torch.Tensor, np.ndarray],
    intervention_values: Union[torch.Tensor, np.ndarray],
    reference_values: Union[torch.Tensor, np.ndarray],
    Ngraphs: int = 1,
    most_likely_graph: bool = False,
) -> np.ndarray:
    """Calculate the quality of real-world testing based on ATE errors between golden standard (AB testing),
    and estimated ATE via treatment assignment policy.

    This is given by equation (7) of the write-up: https://www.overleaf.com/project/626be97093681b20faf29775,
    defined as ATE(do(T)): = E_{graph}E_{Y} [ Y | do(T=intervention_values)]
    - E_{graph}E_{Y}[ Y | do(T=reference_values)]

    This should be calculated using a separate instance of deci object that represents the simulator.

    Args:
        observational_X: the tensor of shape (num_observations, input_dim) containing the observational data
        gathered from the real environment/simulator. Note that there aren't any interventions applied yet.
        treatment_probability: the tensor of  shape (num_obserbations,) containing the probability (induced by
        test policy) that each subject (row) of interventional X is assigned with treatment combinations at
        intervention values.
            hence 1-treatment_probability will be the probability that each subject (row) of interventional X is
            assigned with treatment combinations at reference values.
            Note that the actual treatment for each subject will be sampled and consolidated in this function.
        treatment_mask is a tensor of binary elements of shape (num_observations,) cotaining the simulated
        treatment assighment for each observation.
            treatment_mask_i = 1 indicates that subjet i is assigned with treatment of value intervention_values;
            otherwise reference_values is assigned.
        outcome_idxs: torch Tensor of shape (num_targets) containing indices of variables that specifies the
        outcome Y
        intervention_idxs: torch.Tensor of shape (num_interventions) containing indices of variables that have
        been intervened.
            note that when num_interventions >1, the objective will be calculated as ATE(do(T_1,T_2,...))
        intervention_values: torch.Tensor of shape (num_interventions) optional array containing values for
        variables that have been intervened.
        reference_values: torch.Tensor containing a reference value for the treatment.
        Ngraphs: int containing number of graph samples to draw.
        most_likely_graph: bool indicating whether to deterministically pick the most probable graph under the
        approximate posterior instead of sampling graphs

    Returns:
        ndarray of shape (num_targets,) containing the estimating error for each targets
    """
    ground_truth_value = (
        test_environment.ite(
            X=observational_X,
            intervention_idxs=intervention_idxs,
            intervention_values=intervention_values,
            reference_values=reference_values,
            Ngraphs=Ngraphs,
            most_likely_graph=most_likely_graph,
        )
    )[:, outcome_idxs].mean(axis=0)
    interventional_X = torch.empty_like(torch.tensor(observational_X))
    interventional_X[treatment_mask == 1, :], _ = test_environment.ite(
        observational_X[treatment_mask == 1, :], intervention_idxs, intervention_values, Ngraphs=Ngraphs
    )
    interventional_X[treatment_mask == 0, :], _ = test_environment.ite(
        observational_X[treatment_mask == 0, :], intervention_idxs, reference_values, Ngraphs=Ngraphs
    )
    interventional_X += observational_X
    ipw_estimated_value = get_ipw_estimated_ate(interventional_X, treatment_probability, treatment_mask, outcome_idxs)

    return (ground_truth_value - ipw_estimated_value) ** 2


def organize_temporal_discovery_results(
    adj_metrics_all_full: Optional[dict] = None,
    adj_metrics_all_temp: Optional[dict] = None,
    adj_metrics_inst: Optional[dict] = None,
    adj_metrics_lag: Optional[dict] = None,
    adj_metrics_agg: Optional[dict] = None,
) -> dict:
    """
    This will aggregate the causal discovery results into a single result dict.
    Args:
        adj_metrics_all_full: The results for full-graph discovery
        adj_metrics_all_temp: The results for temporal graph discovery
        adj_metrics_inst: The results for temporal graph discovery with just instantaneous effect.
        adj_metrics_lag: The results for temporal graph discovery with just lagged effect.
        adj_metrics_agg: The results for aggregated graph discovery.

    Returns:
        adj_results: results dict aggregating the above metrics.
    """
    adj_results = {}
    if adj_metrics_all_full is not None:
        for key, value in adj_metrics_all_full.items():
            adj_results[f"{key}_overall_full_time"] = value

    if adj_metrics_all_temp is not None:
        for key, value in adj_metrics_all_temp.items():
            adj_results[f"{key}_overall_temporal"] = value

    if adj_metrics_inst is not None:
        for key, value in adj_metrics_inst.items():
            adj_results[f"{key}_inst"] = value

    if adj_metrics_lag is not None:
        for key, value in adj_metrics_lag.items():
            adj_results[f"{key}_lag"] = value

    if adj_metrics_agg is not None:
        for key, value in adj_metrics_agg.items():
            adj_results[f"{key}_agg"] = value

    return adj_results


def eval_temporal_causal_discovery(
    dataset: TemporalDataset, model: IModelForCausalInference, disable_diagonal_eval: bool = True
) -> dict:
    """
    This will evaluate the temporal causal discovery performance. It includes 4 different comparisons:
    (1) full-time graph with inst and lag; (2) temporal graph with inst and lag; (3) temporal graph with just inst;
    (4) temporal graph with just inst.

    Args:
        dataset: The dataset used which contains the ground truth temporal adj matrix.
        model: the trained model we want to evaluate.
        disable_diagonal_eval: whether to disable the diagonal elements in aggregation for evaluation

    Returns:
        adj_metrics: dict containing the discovery results
    """
    adj_ground_truth = dataset.get_adjacency_data_matrix()

    # For DECI-based model, the default is to give 100 samples of the graph posterior
    adj_pred = model.get_adj_matrix().astype(float).round()
    adj_agg_metrics = {}
    if model.name() in ("rhino", "varlingam", "dynotears", "pcmci_plus"):
        if adj_ground_truth.ndim == 3:
            # temporal adj matrix
            adj_ground_truth, adj_pred = make_temporal_adj_matrix_compatible(
                adj_ground_truth, adj_pred, is_static=False
            )  # [maxlag+1, num_nodes, num_nodes], [batch, maxlag+1, num_nodes, num_nodes]
            for name_dict in ["all_full", "all_temp", "inst", "lag"]:
                results = evaluate_temporal_causal_discovery_subtype(
                    adj_ground_truth=adj_ground_truth, adj_pred=adj_pred, subtype=name_dict
                )

                adj_agg_metrics["adj_metrics_" + name_dict] = results
        elif adj_ground_truth.ndim == 2:
            # aggregated adj matrix
            # The true adj is aggregated
            if adj_pred.ndim == 3:
                adj_pred = adj_pred[np.newaxis, ...]  # [1, lag+1, num_nodes, num_nodes]

            adj_matrix = (adj_pred.sum(1) > 0).astype(int)  # [batch, num_node, num_node]
            # get bernoulli prob for auroc computation
            if model.name() == "rhino":
                adj_pred_prob = model.get_adj_matrix(do_round=False, samples=1, most_likely_graph=True)[0].max(
                    0
                )  # [num_nodes, num_nodes]
            else:
                adj_pred_prob = adj_matrix.mean(0)  # [num_nodes, num_nodes]

            # disable diagonal
            if disable_diagonal_eval:
                for cur_adj_matrix in adj_matrix:
                    np.fill_diagonal(cur_adj_matrix, 0)
                np.fill_diagonal(adj_pred_prob, 0)
            # evaluate the metrics
            results = edge_prediction_metrics_multisample(
                adj_ground_truth,
                adj_matrix,
                adj_matrix_mask=None,
                compute_mean=True,
                adj_pred_prob=adj_pred_prob,
            )
            # Add FPR, TPR
            adj_matrix_true_fl = adj_ground_truth.flatten()
            adj_pred_fl = adj_pred_prob.flatten()
            FPR, TPR, _ = metrics.roc_curve(adj_matrix_true_fl, adj_pred_fl)
            results["FPR"] = list(FPR)  # type: ignore
            results["TPR"] = list(TPR)  # type: ignore
            adj_agg_metrics["adj_metrics_agg"] = results

    elif model.name() == "fold_time_deci":
        adj_ground_truth, adj_pred = make_temporal_adj_matrix_compatible(
            adj_ground_truth, adj_pred, is_static=True, adj_matrix_2_lag=model.lag  # type: ignore
        )
        results = evaluate_temporal_causal_discovery_subtype(
            adj_ground_truth=adj_ground_truth, adj_pred=adj_pred, subtype="all_full", is_static=True
        )

        adj_agg_metrics["adj_metrics_all_full"] = results

    adj_results = organize_temporal_discovery_results(**adj_agg_metrics)
    return adj_results


def evaluate_temporal_causal_discovery_subtype(
    adj_ground_truth: np.ndarray, adj_pred: np.ndarray, subtype: str, is_static: bool = False
) -> Dict[str, float]:
    """
    This will evaluate the temporal causal discovery with a specific evaluation type. Currently, it supports
        (1) "adj_metrics_all_full": the full-time graph with both inst and lagged effect; (2) "adj_metrics_all_temp":
        the temporal graph with both inst and lagged effect; (3) "adj_metrics_inst": the temporal graph with just inst;
        (4) "adj_metrics_lag": the temporal graph with just lagged effect.

    Args:
        adj_ground_truth: The ground truth temporal adj matrix with shape [lag+1, num_nodes, num_nodes]. It should have the
            compatible lags with adj_pred.
        adj_pred: The sampled adj matrix from the model with shape either [batch, lag+1, num_nodes, num_nodes] or
            [model_lag+1, num_nodes, num_nodes] for temporal model. For static temporal model, it should be [(lag+1)*num_nodes, (lag+1)*num_nodes] or
            [batch, (lag+1)*num_nodes, (lag+1)*num_nodes]. It should have the compatible lags with adj_ground_truth.
        subtype: the type of evaluation to perform. It supports "all_full", "all_temp", "inst"
            and "lag". (1) "all_full": It will convert the temporal graph ([lat+1, num_nodes, num_nodes]) to full-time
            static graph ([(lag+1)*num_nodes, (lag+1)*num_nodes]); (2) "all_temp": Same as "all_full" but now convert to
            static version of temporal graph, not full-time graph; (3) "inst": Extract the instantaneous adj matrix from temporal adj matrix;
            (4) "lag": It will convert the lag part of temporal adj matrix to its static version similar to "all_temp".
        is_static: whether the adj_pred matrix generated by a temporal model based on static discovery method. E.g. fold-time DECI.
    Returns:
        adj_metrics: dict containing the discovery results
    """

    assert subtype in [
        "all_full",
        "all_temp",
        "inst",
        "lag",
    ], "Unknown subtype"
    if is_static:
        assert subtype == "all_full", "For static temporal model, only full-time graph is supported."
        assert adj_pred.ndim in (2, 3)
        assert (
            adj_pred.shape[-1] // adj_ground_truth.shape[-1] == adj_ground_truth.shape[0]
        ), "adj_pred should have a compatible lag with adj_ground_truth"
    else:
        assert adj_pred.ndim in (3, 4)
        assert (
            adj_pred.shape[-3] == adj_ground_truth.shape[0]
        ), "adj_pred should have a compatible lag with adj_ground_truth"

    if subtype == "all_full":
        # full-time graph with both inst and lag
        cur_adj_true = convert_temporal_to_static_adjacency_matrix(adj_ground_truth, conversion_type="full_time")
        if not is_static:
            cur_adj_pred = convert_temporal_to_static_adjacency_matrix(adj_pred, conversion_type="full_time")
        else:
            cur_adj_pred = adj_pred
    elif subtype == "all_temp":
        # temporal graph with inst and lag
        cur_adj_true = convert_temporal_to_static_adjacency_matrix(adj_ground_truth, conversion_type="auto_regressive")
        cur_adj_pred = convert_temporal_to_static_adjacency_matrix(adj_pred, conversion_type="auto_regressive")
    elif subtype == "inst":
        # temporal graph wth just inst
        cur_adj_true = adj_ground_truth[0]
        cur_adj_pred = adj_pred[..., 0, :, :]  # [num_nodes, num_nodes] or [batch, num_nodes, num_nodes]
    elif subtype == "lag":
        # temporal graph with just lag
        cur_adj_true = convert_temporal_to_static_adjacency_matrix(
            adj_ground_truth[1:, :, :], conversion_type="auto_regressive"
        )
        cur_adj_pred = convert_temporal_to_static_adjacency_matrix(
            adj_pred[..., 1:, :, :], conversion_type="auto_regressive"
        )

    if len(cur_adj_pred.shape) == 2:
        # If predicts single adjacency matrix
        results = edge_prediction_metrics(cur_adj_true, cur_adj_pred, adj_matrix_mask=None)
    elif len(cur_adj_pred.shape) == 3:
        # If predicts multiple adjacency matrices (stacked)
        results = edge_prediction_metrics_multisample(cur_adj_true, cur_adj_pred, adj_matrix_mask=None)

    return results
