from typing import Dict

import torch

from .context_treatment import BayesianContextTreatmentModel, Parameters


class SingleConfoundingRoot(BayesianContextTreatmentModel):
    """
    Single confounder treatment-effect fixed graph, i.e. 𝑐 causes 𝑎 and 𝑦, 𝑎 causes 𝑦.

    We don't place priors on 𝑎 and 𝑐. Although in the fully Bayesian single confounding root scenario
    𝑎 = 𝑓(𝑐) + 𝜀, in the experimental design context we always choose 𝑎, i.e. we do(𝑎), and so learning
    a distribution for 𝑎 is not necessary.
    """

    def __init__(
        self, priors_on_parameters: Dict, treatment_policy: torch.nn.Module, return_max_reward: bool = True
    ) -> None:
        super().__init__(
            priors_on_parameters=priors_on_parameters,
            treatment_policy=treatment_policy,
            return_max_reward=return_max_reward,
        )

    def sample_interventional_data(
        self,
        parameters: Parameters,
        context: torch.Tensor,
        effect_variable: str = "y",
    ):
        return self.sample_observational_data(parameters, context, effect_variable)

    def sample_counterfactual_data(
        self,
        parameters: Parameters,
        context: torch.Tensor,
        effect_variable: str,
        factual_outcome,
    ) -> torch.Tensor:
        raise NotImplementedError()
