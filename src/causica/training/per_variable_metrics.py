from typing import Callable, Iterable, Optional

import torch
from tensordict import TensorDict

from causica.datasets.causica_dataset_format import CounterfactualWithEffects
from causica.datasets.tensordict_utils import expand_tensordict_groups
from causica.distributions.transforms import JointTransform
from causica.sem.distribution_parameters_sem import DistributionParametersSEM
from causica.sem.structural_equation_model import SEM, counterfactual


def calculate_counterfactual_deci_metrics(
    sems: Iterable[SEM],
    counterfactual_data: CounterfactualWithEffects,
    grouped_variable_names: Optional[dict[str, list[str]]] = None,
    standardizer: Optional[JointTransform] = None,
) -> dict[str, TensorDict]:
    """Evaluate the counterfacual rmses of a model.

    Args:
        sems: An iterable of structural equation models to evaluate the ITE RMSE of
        counterfactual_data: Data of true counterfactuals to use for evaluation.
        grouped_variable_names: Optional dictionary that holds the names of the variables in each group. If given,
            the variables are evaluated individually. Otherwise, the variables are evaluated groupwise.
        standardizer: Standardizer that is used to invert the predictions and loaded data to convery the metrics into
            the space of the original data.

    Returns:
        Dict of RMSEs and MAPEs for each effect variable we're interested in
    """
    intervention, _, effects = counterfactual_data

    # generate samples from the intervened distribution and the base distribution
    stacked: TensorDict = torch.stack(
        [counterfactual(sem, intervention.factual_data, intervention.intervention_values) for sem in sems]
    )
    generated_cf_outcomes = stacked.apply(
        lambda v: v.mean(axis=0), batch_size=intervention.factual_data.batch_size, inplace=False
    )
    true_counterfactual_outcome = intervention.counterfactual_data

    if standardizer is not None:
        generated_cf_outcomes = standardizer.inv(generated_cf_outcomes)
        true_counterfactual_outcome = standardizer.inv(true_counterfactual_outcome)

    if grouped_variable_names:
        generated_cf_outcomes = expand_tensordict_groups(generated_cf_outcomes, grouped_variable_names)
        true_counterfactual_outcome = expand_tensordict_groups(true_counterfactual_outcome, grouped_variable_names)
        effects = {
            var for group_name, group in grouped_variable_names.items() for var in group if group_name in effects
        }

    return {
        "rmse": eval_per_variable_metric(generated_cf_outcomes, true_counterfactual_outcome, effects, rmse),
        "mape": eval_per_variable_metric(generated_cf_outcomes, true_counterfactual_outcome, effects, mape),
        "smape": eval_per_variable_metric(generated_cf_outcomes, true_counterfactual_outcome, effects, smape),
    }


def calculate_observational_deci_metrics(
    sems: Iterable[DistributionParametersSEM],
    observations: TensorDict,
    continuous_variables: list[str],  # keys to variables to be interpreted as categorical
    binary_variables: list[str],  # keys to variables to be interpreted as binary
    categorical_variables: list[str],  # keys to variables to be interpreted as categorical
    grouped_variable_names: Optional[dict[str, list[str]]] = None,
    standardizer: Optional[JointTransform] = None,
) -> dict[str, TensorDict]:
    """Calculates the RMSE and accuracy of the predictions of a model.

    Args:
        sems: An iterable of structural equation models to evaluate the ITE RMSE of
        observations: Observational data to evaluate
        continuous_variables: Keys of the continuous variables
        binary_variables: Keys of the binary variables
        categorical_variables: Keys of the categorical variables
        grouped_variable_names: Optional dictionary that holds the names of the variables in each group. If given,
            the variables are evaluated individually. Otherwise, the variables are evaluated groupwise.
        standardizer: Standardizer that is used to invert the predictions and loaded data to convery the metrics into
            the space of the original data.

    Returns:
        Dict holding the different metrics
    """
    assert all(
        set(sem.node_names) == set(observations.keys()) for sem in sems
    ), f"observations must be compatible with SEMs: got {set(observations.keys())} but expected {set(list(sems)[0].node_names)}"

    assert set(continuous_variables + binary_variables + categorical_variables) == set(
        observations.keys()
    ), f"observations must be compatible with variables: got {set(observations.keys())} but expected {set(continuous_variables + binary_variables + categorical_variables)}"
    assert set(continuous_variables).isdisjoint(
        set(binary_variables)
    ), f"continuous and binary variables must be disjoint: got {set(continuous_variables).intersection(set(binary_variables))}"
    assert set(continuous_variables).isdisjoint(
        set(categorical_variables)
    ), f"continuous and categorical variables must be disjoint: got {set(continuous_variables).intersection(set(categorical_variables))}"
    assert set(binary_variables).isdisjoint(
        set(categorical_variables)
    ), f"binary and categorical variables must be disjoint: got {set(binary_variables).intersection(set(categorical_variables))}"

    stacked: TensorDict = torch.stack([sem.func(observations, sem.graph) for sem in sems])
    mean_predictions = stacked.apply(lambda v: v.mean(axis=0), batch_size=observations.batch_size, inplace=False)

    if standardizer is not None:
        mean_predictions = standardizer.inv(mean_predictions)
        observations = standardizer.inv(observations)

    if grouped_variable_names:
        mean_predictions = expand_tensordict_groups(mean_predictions, grouped_variable_names)
        observations = expand_tensordict_groups(observations, grouped_variable_names)
        continuous_variables = [
            var for group in grouped_variable_names.values() for var in group if var in continuous_variables
        ]
        binary_variables = [
            var for group in grouped_variable_names.values() for var in group if var in binary_variables
        ]
        categorical_variables = [
            var for group in grouped_variable_names.values() for var in group if var in categorical_variables
        ]

    return {
        "rmse": eval_per_variable_metric(mean_predictions, observations, continuous_variables, rmse),
        "mape": eval_per_variable_metric(mean_predictions, observations, continuous_variables, mape),
        "smape": eval_per_variable_metric(mean_predictions, observations, continuous_variables, smape),
        "binary_accuracy": eval_per_variable_metric(mean_predictions, observations, binary_variables, binary_accuracy),
        "categorical_accuracy": eval_per_variable_metric(
            mean_predictions, observations, categorical_variables, categorical_accuracy
        ),
    }


def eval_per_variable_metric(
    predictions: TensorDict,
    observations: TensorDict,
    variables: Iterable[str],
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> TensorDict:
    """Calculates a metric for  and accuracy of the predictions of a model.

    Args:
        predictions: Predictions to evaluate
        observations: Observational data to evaluate
        variables: Keys of the variables to evaluate the metric for
        metric: Metric to evaluate
        grouped_variable_names: Optional dictionary that holds the names of the variables in each group. If given,
            the variables are evaluated individually. Otherwise, the variables are evaluated groupwise.
        standardizer: Standardizer that is used to invert the predictions and loaded data to convery the metrics into
            the space of the original data.

    Returns:
        Dict holding the metric for each node
    """
    assert set(variables).issubset(
        predictions.keys()
    ), f"predictions must contain all variables: got {predictions.keys()} but expected {variables}"
    assert set(variables).issubset(
        observations.keys()
    ), f"observations must contain all variables: got {observations.keys()} but expected {variables}"

    metric_results = predictions.select(*variables).apply(
        metric, observations.select(*variables), batch_size=torch.Size([])
    )

    return metric_results


def binary_accuracy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate the accuracy of a prediction for a binary variable.

    Args:
        logits: Tensor of logits [batch_size] or [batch_size, num_dims_for_node]
        target: Tensor of targets [batch_size] or [batch_size, num_dims_for_node]

    Returns:
        Accuracy of the prediction
    """
    return torch.mean(((torch.sigmoid(logits) > 0.5) == target).float())


def categorical_accuracy(logits: torch.Tensor, target: torch.Tensor, target_is_onehot: bool = True) -> torch.Tensor:
    """Calculate the accuracy of a prediction for a categorical variable.

    Args:
        logits: Tensor of logits [batch_size, num_classes]
        target: Tensor of targets [batch_size, num_classes] (if target_is_onehot is True) or [batch_size] (otherwise)
        target_is_onehot: Whether the target is one hot. Defaults to True.

    Returns:
        Accuracy of the prediction
    """
    if target_is_onehot:
        target = torch.argmax(target, -1)

    prediction = torch.argmax(logits, dim=-1)

    return torch.mean((prediction == target).float())


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate the RMSE of a prediction. This will sum over all dimensions except the batch dimension.

    Args:
        prediction: Tensor of predictions [batch_size, num_dims_for_node]
        target: Tensor of targets [batch_size, num_dims_for_node]

    Returns:
        RMSE of the prediction
    """
    return torch.sqrt(torch.mean(torch.sum((prediction - target) ** 2, -1)))


def mape(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate the mean absolute percentage error of a prediction. This will sum over all dimensions except the batch dimension.

    Args:
        prediction: Tensor of predictions [batch_size, num_dims_for_node]
        target: Tensor of targets [batch_size, num_dims_for_node]

    Returns:
        MAPE of the prediction
    """

    return torch.mean(torch.nansum(torch.abs((prediction - target) / target), -1))


def smape(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate the symmetric mean absolute percentage error of a prediction. This will sum over all dimensions except the batch dimension.

    Args:
        prediction: Tensor of predictions [batch_size, num_dims_for_node]
        target: Tensor of targets [batch_size, num_dims_for_node]

    Returns:
        sMAPE of the prediction
    """
    return torch.mean(
        torch.sum(
            torch.where(
                torch.abs(target) + torch.abs(prediction) == 0,
                torch.zeros_like(target),
                torch.abs((prediction - target) / (torch.abs(target) + torch.abs(prediction))),
            ),
            -1,
        )
    )
