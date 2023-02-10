import math
from collections import defaultdict
from itertools import chain
from typing import Dict, Iterable, List

import torch

from causica.datasets.csuite_data import CounterfactualsWithEffects, InterventionsWithEffects
from causica.sem.structural_equation_model import SEM, ate, ite


def eval_intervention_likelihoods(sems: List[SEM], interventions: InterventionsWithEffects) -> torch.Tensor:
    """
    Calculate the average log-prob of interventional data.

    Specifically we calculate 𝔼_sample[log(𝔼_G[p(sample | G)])]

    Args:
        sems: An iterable of SEMS to evaluate the interventional log prob of
        interventions (InterventionsWithEffects): True interventional data to use for evaluation.

    Returns:
        float: Log-likelihood of the interventional data.
    """
    interventions_iterator = chain.from_iterable((a, b) for a, b, _ in interventions)
    total_log_sum_exp_per_int = []  # this will end up being length number of interventions
    num_samples = 0  # keep track of the total number of samples
    for intervention in interventions_iterator:
        inner_log_probs = [
            sem.do(interventions=intervention.intervention_values).log_prob(intervention.intervention_data)
            for sem in sems
        ]
        # calculate log(Σ_G p(sample | G)) for each sample
        log_prob = list_logsumexp(inner_log_probs)  # batch_size
        assert len(log_prob.shape) == 1
        num_samples += log_prob.shape[-1]
        total_log_sum_exp_per_int.append(log_prob.sum())

    # Σ_sample log(Σ_G p(sample | G))
    total_log_prob = torch.stack(total_log_sum_exp_per_int, dim=0).sum(dim=0)
    # take the average over the datapoints and divide this by the number of graphs (inside the log)
    return total_log_prob / num_samples - math.log(len(sems))


def eval_ate_rmse(
    sems: Iterable[SEM], interventions: InterventionsWithEffects, samples_per_graph: int = 1000
) -> torch.Tensor:
    """Evaluate the ATEs of a model

    Args:
        sems: An iterable of structural equation models to evaluate the ATE RMSE of
        interventions (InterventionsWithEffects): True interventional data to use for evaluation.
        samples_per_graph (int, optional): Number of samples to draw per graph to calculate the ATE. Defaults to 1000.

    Returns:
        float: RMSE of the ATEs.
    """

    ate_rmses = []
    for (intervention_a, intervention_b, effects) in interventions:

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

        generated_ates = {k: list_mean(v) for k, v in ates_per_graph.items()}

        ate_rmse = torch.sqrt(list_mean([torch.sum((generated_ates[key] - true_ates[key]) ** 2) for key in effects]))
        ate_rmses.append(ate_rmse)
    return list_mean(ate_rmses)


def eval_ite_rmse(sems: Iterable[SEM], counterfactuals: CounterfactualsWithEffects) -> torch.Tensor:
    """Evaluate the ITEs of a model.

    Args:
        sems: An iterable of structural equation models to evaluate the ITE RMSE of
        counterfactuals (CounterfactualsWithEffects): Data of true counterfactuals to use for evaluation.

    Returns:
        float: RMSE of the ITEs.
    """
    ite_rmses = []
    for (intervention_a, intervention_b, effects) in counterfactuals:
        for key, a_val in intervention_a.factual_data.items():
            b_val = intervention_b.factual_data[key]
            assert torch.allclose(a_val, b_val), "Base data must be the same for ITEs"
        true_ites = {
            effect: intervention_a.counterfactual_data[effect] - intervention_b.counterfactual_data[effect]
            for effect in effects
        }

        factual_data_dict = intervention_a.factual_data

        # generate samples from the intervened distribution and the base distribution
        per_graph_ites: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for sem in sems:
            sem_ites = ite(
                sem, factual_data_dict, intervention_a.intervention_values, intervention_b.intervention_values, effects
            )

            for key in effects:
                per_graph_ites[key].append(sem_ites[key].detach())

        generated_ites = {k: list_mean(v) for k, v in per_graph_ites.items()}

        ite_rmse = torch.sqrt(
            list_mean([torch.sum((generated_ites[key] - true_ites[key]) ** 2, -1) for key in effects])
        )
        ite_rmses.append(ite_rmse)

    return torch.mean(list_mean(ite_rmses))


def list_mean(list_of_tensors: List[torch.Tensor]) -> torch.Tensor:
    """Take the mean of a list of torch tensors, they must all have the same shape"""
    return torch.stack(list_of_tensors, dim=0).mean(dim=0)


def list_logsumexp(list_of_tensors: List[torch.Tensor]) -> torch.Tensor:
    """Take the logsumexp of a list of torch tensors, they must all have the same shape"""
    return torch.logsumexp(torch.stack(list_of_tensors, dim=0), dim=0)
