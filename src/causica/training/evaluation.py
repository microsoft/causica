import math
from collections import defaultdict
from typing import Iterable

import torch
from tensordict import TensorDict

from causica.datasets.causica_dataset_format import CounterfactualWithEffects, InterventionWithEffects
from causica.sem.structural_equation_model import SEM, ate, ite


def eval_intervention_likelihoods(sems: list[SEM], intervention_with_effects: InterventionWithEffects) -> torch.Tensor:
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
    ates_per_graph: dict[str, list[torch.Tensor]] = defaultdict(list)
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


def eval_ite_rmse(sems: Iterable[SEM], counterfactual_data: CounterfactualWithEffects) -> TensorDict:
    """Evaluate the ITEs of a model.

    Args:
        sems: An iterable of structural equation models to evaluate the ITE RMSE of
        counterfactual_data: Data of true counterfactuals to use for evaluation.

    Returns:
        Dict of RMSEs for each effect variable we're interested in
    """
    intervention_a, intervention_b, effects = counterfactual_data
    if intervention_b is None:
        raise ValueError("ITE evaluation must have reference counterfactuals")
    for key, a_val in intervention_a.factual_data.items():
        b_val = intervention_b.factual_data[key]
        assert torch.allclose(a_val, b_val), "Base data must be the same for ITEs"
    # each node has shape [batch_size, node_dim]
    true_ites = {
        effect: intervention_a.counterfactual_data[effect] - intervention_b.counterfactual_data[effect]
        for effect in effects
    }

    # generate samples from the intervened distribution and the base distribution
    per_graph_ites: dict[str, list[torch.Tensor]] = defaultdict(list)
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


def list_mean(list_of_tensors: list[torch.Tensor]) -> torch.Tensor:
    """Take the mean of a list of torch tensors, they must all have the same shape"""
    return torch.stack(list_of_tensors, dim=0).mean(dim=0)


def list_logsumexp(list_of_tensors: list[torch.Tensor]) -> torch.Tensor:
    """Take the logsumexp of a list of torch tensors, they must all have the same shape"""
    return torch.logsumexp(torch.stack(list_of_tensors, dim=0), dim=0)
