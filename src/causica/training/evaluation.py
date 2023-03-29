import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Union

import torch
from tensordict import TensorDict

from causica.datasets.causica_dataset_format import CounterfactualWithEffects, InterventionWithEffects
from causica.sem.distribution_parameters_sem import DistributionParametersSEM
from causica.sem.structural_equation_model import SEM, ate, ite


def eval_intervention_likelihoods(sems: List[SEM], intervention_with_effects: InterventionWithEffects) -> torch.Tensor:
    """
    Calculate the average log-prob of interventional data.

    Specifically we calculate 𝔼_sample[log(𝔼_G[p(sample | G)])]

    Args:
        sems: An iterable of SEMS to evaluate the interventional log prob of
        interventions: True interventional data to use for evaluation.

    Returns:
        Log-likelihood of the interventional data for each interventional datapoint
    """
    total_log_sum_exp_per_int = []  # this will end up being length number of interventions
    for intervention in intervention_with_effects[:2]:
        inner_log_probs = [
            sem.do(interventions=intervention.intervention_values).log_prob(intervention.intervention_data)
            for sem in sems
        ]
        # calculate log(𝔼_G[p(sample | G)]) for each sample
        log_prob = list_logsumexp(inner_log_probs) - math.log(len(sems))  # batch_size
        assert len(log_prob.shape) == 1
        total_log_sum_exp_per_int.append(log_prob)

    # log(𝔼_G[p(sample | G)]) for each sample in both interventions, shape 2 * batch_size
    return torch.cat(total_log_sum_exp_per_int, dim=-1)


def eval_ate_rmse(
    sems: Iterable[SEM], intervention: InterventionWithEffects, samples_per_graph: int = 1000
) -> TensorDict:
    """Evaluate the ATEs of a model

    Args:
        sems: An iterable of structural equation models to evaluate the ATE RMSE of
        intervention: True interventional data to use for evaluation.
        samples_per_graph: Number of samples to draw per graph to calculate the ATE.

    Returns:
        Dict of the RMSE of the ATE for each node we're interested in
    """

    intervention_a, intervention_b, effects = intervention

    # each node has shape [batch_size, node_dim]
    true_ates = {
        effect: intervention_a.intervention_data[effect].mean() - intervention_b.intervention_data[effect].mean()
        for effect in effects
    }

    # generate samples from the intervened distribution and the base distribution
    ates_per_graph: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for sem in sems:
        graph_ates = ate(
            sem, intervention_a.intervention_values, intervention_b.intervention_values, effects, samples_per_graph
        )
        for key in effects:
            ates_per_graph[key].append(graph_ates[key].detach())

    # each node has shape [node_dim]
    generated_ates = {k: list_mean(v) for k, v in ates_per_graph.items()}
    return TensorDict(
        {key: torch.sqrt(torch.mean(torch.sum((generated_ates[key] - true_ates[key]) ** 2, -1))) for key in effects},
        batch_size=torch.Size([]),
    )


def eval_ite_rmse(sems: Iterable[SEM], counterfactual: CounterfactualWithEffects) -> TensorDict:
    """Evaluate the ITEs of a model.

    Args:
        sems: An iterable of structural equation models to evaluate the ITE RMSE of
        counterfactual: Data of true counterfactuals to use for evaluation.

    Returns:
        Dict of RMSEs for each effect variable we're interested in
    """
    intervention_a, intervention_b, effects = counterfactual
    for key, a_val in intervention_a.factual_data.items():
        b_val = intervention_b.factual_data[key]
        assert torch.allclose(a_val, b_val), "Base data must be the same for ITEs"
    # each node has shape [batch_size, node_dim]
    true_ites = {
        effect: intervention_a.counterfactual_data[effect] - intervention_b.counterfactual_data[effect]
        for effect in effects
    }

    # generate samples from the intervened distribution and the base distribution
    per_graph_ites: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for sem in sems:
        sem_ites = ite(
            sem,
            intervention_a.factual_data,
            intervention_a.intervention_values,
            intervention_b.intervention_values,
            effects,
        )

        for key in effects:
            per_graph_ites[key].append(sem_ites[key].detach())

    # average the treatment value over all graphs, each node has shape [batch_size, node_dim]
    generated_ites = {k: list_mean(v) for k, v in per_graph_ites.items()}
    return TensorDict(
        {key: torch.sqrt(torch.mean(torch.sum((generated_ites[key] - true_ites[key]) ** 2, -1))) for key in effects},
        batch_size=torch.Size(),
    )


def eval_observational_per_variable_rmse_and_accuracy(
    sems: Union[List[DistributionParametersSEM], Tuple[DistributionParametersSEM]],
    observations: TensorDict,
    continuous_variables: List[str],  # keys to variables to be interpreted as categorical
    binary_variables: List[str],  # keys to variables to be interpreted as binary
    categorical_variables: List[str],  # keys to variables to be interpreted as categorical
) -> Dict[str, torch.Tensor]:
    """Calculates the RMSE and accuracy of the predictions of a model.

    Args:
        sems: An iterable of structural equation models to evaluate the ITE RMSE of
        observations: Observational data to evaluate

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
    mean_predictions = stacked.apply(lambda v: v.mean(axis=0), batch_size=(len(sems),), inplace=False)

    metrics = {}
    for var in continuous_variables:
        metrics[f"rmse_{var}"] = rmse(mean_predictions.get(var), observations.get(var))

    for var in binary_variables:
        metrics[f"accuracy_{var}"] = binary_accuracy(mean_predictions.get(var), observations.get(var))

    for var in categorical_variables:
        metrics[f"accuracy_{var}"] = categorical_accuracy(mean_predictions.get(var), observations.get(var))

    return metrics


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


def list_mean(list_of_tensors: List[torch.Tensor]) -> torch.Tensor:
    """Take the mean of a list of torch tensors, they must all have the same shape"""
    return torch.stack(list_of_tensors, dim=0).mean(dim=0)


def list_logsumexp(list_of_tensors: List[torch.Tensor]) -> torch.Tensor:
    """Take the logsumexp of a list of torch tensors, they must all have the same shape"""
    return torch.logsumexp(torch.stack(list_of_tensors, dim=0), dim=0)
