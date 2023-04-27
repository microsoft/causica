from typing import Iterable, Optional, Union

import torch
from tensordict import TensorDict

from causica.datasets.causica_dataset_format import CounterfactualWithEffects
from causica.distributions.transforms import JointTransform
from causica.sem.distribution_parameters_sem import DistributionParametersSEM
from causica.sem.structural_equation_model import SEM, counterfactual


def eval_counterfactual_outcome_per_variable_rmse(
    sems: Iterable[SEM],
    counterfactual_data: CounterfactualWithEffects,
    grouped_variable_names: Optional[dict[str, list[str]]] = None,
    standardizer: Optional[JointTransform] = None,
) -> TensorDict:
    """Evaluate the counterfacual rmses of a model.

    Args:
        sems: An iterable of structural equation models to evaluate the ITE RMSE of
        counterfactual_data: Data of true counterfactuals to use for evaluation.
        grouped_variable_names: Optional dictionary that holds the names of the variables in each group. If given,
            the variables are evaluated individually. Otherwise, the variables are evaluated groupwise.
        standardizer: Standardizer that is used to invert the predictions and loaded data to convery the metrics into
            the space of the original data.

    Returns:
        Dict of RMSEs for each effect variable we're interested in
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

    if grouped_variable_names is None:
        # calculate the rmse metrics, one for each group
        rmses = TensorDict(
            {key: rmse(generated_cf_outcomes[key], true_counterfactual_outcome[key]) for key in effects},
            batch_size=torch.Size(),
        )
    else:
        # calculate the rmse metrics, one for each column/single variable
        rmses = TensorDict(
            {
                col_name: rmse(generated_cf_outcomes[key][:, idx], true_counterfactual_outcome[key][:, idx])
                for key in effects
                for idx, col_name in enumerate(grouped_variable_names[key])
            },
            batch_size=torch.Size(),
        )

    return rmses


def eval_observational_per_variable_rmse_and_accuracy(
    sems: Union[list[DistributionParametersSEM], tuple[DistributionParametersSEM]],
    observations: TensorDict,
    continuous_variables: list[str],  # keys to variables to be interpreted as categorical
    binary_variables: list[str],  # keys to variables to be interpreted as binary
    categorical_variables: list[str],  # keys to variables to be interpreted as categorical
    grouped_variable_names: Optional[dict[str, list[str]]] = None,
    standardizer: Optional[JointTransform] = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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
        Dict holding the accuracy metrics for each node
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

    rmses = {}
    accuraccies = {}
    if grouped_variable_names is None:
        # Calculate metrics for each group
        for var in continuous_variables:
            rmses[f"{var}"] = rmse(mean_predictions.get(var), observations.get(var))

        for var in binary_variables:
            accuraccies[f"{var}"] = binary_accuracy(mean_predictions.get(var), observations.get(var))

        for var in categorical_variables:
            accuraccies[f"{var}"] = categorical_accuracy(mean_predictions.get(var), observations.get(var))
    else:
        # Calculate metrics for each variable in each group
        for group_name in continuous_variables:
            for i, var in enumerate(grouped_variable_names[group_name]):
                rmses[f"{var}"] = rmse(
                    mean_predictions.get(group_name)[:, i][:, None], observations.get(group_name)[:, i][:, None]
                )

        for group_name in binary_variables:
            for i, var in enumerate(grouped_variable_names[group_name]):
                accuraccies[f"{var}"] = binary_accuracy(
                    mean_predictions.get(group_name)[:, i], observations.get(group_name)[:, i]
                )

        # Categorical variables are one-hot encoded, so we need to calculate the accuracy for each group as we do not
        # support grouped categorical variables of shape [batch_size, num_groups, num_dims_for_node]
        for group_name in categorical_variables:
            accuraccies[f"{group_name}"] = categorical_accuracy(
                mean_predictions.get(group_name), observations.get(group_name)
            )

    return rmses, accuraccies


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
