from typing import Dict, Optional, Tuple

import pyro
import pyro.distributions as dist
import torch

from .single_confounding_root import Parameters, SingleConfoundingRoot


class ContinuousContextBinaryTreatment(SingleConfoundingRoot):
    """
    This is a single confounding root Bayesian toy data simulator of the form 𝑦 = 𝑓(𝑎, 𝑐) + 𝜎N(0, 1) where
    𝑎 = [1, 0] or [0, 1] is a one hot treatment, set by the user, 𝑐 is a 1d context, which is given,
    𝑓 is a random function, parameterised by 𝛙₀, 𝛙₁ ~ N(0, I), shape [2, latent_dim] as follows:

       𝑓(𝑎, 𝑐;𝛙) = tanh(𝑐 (𝛙₀𝑎)) . (𝛙₁𝑎)

    Note:
        We don't place priors on 𝑎 and 𝑐. Although in the fully Bayesian single confounding root scenario
        𝑎 = 𝑓(𝑐) + 𝜀, in the experimental design context we always choose 𝑎, i.e. we do(𝑎), and so learning
        a distribution for 𝑎 is not necessary.
    """

    TREATMENT_DIM = 2

    def __init__(
        self,
        treatment_policy: torch.nn.Module,
        priors_on_parameters: Optional[Dict] = None,
        noise_scale: float = 0.1,
        latent_dim: int = 4,
    ) -> None:
        """
        Args:
            priors_on_parameters: dict of length 2 containing the priors (pyro.distributions).
                It should be in the form {parameter_name: pyro.distributions}.
                If unspecified (default), the prior is set to be independent N(0, I) for 𝛙₀, 𝛙₁
            noise_scale: Standard deviation (scale) of the noise, 𝜎
        """
        if priors_on_parameters is None:
            priors_on_parameters = {
                f"psi_{i}": dist.MultivariateNormal(
                    torch.zeros((1, self.TREATMENT_DIM, latent_dim)),
                    torch.eye(latent_dim),
                ).to_event(2)
                for i in range(2)  # 2 layer NN
            }
        elif len(priors_on_parameters) != 2:
            raise ValueError(f"len(priors_on_parameters)={len(priors_on_parameters)}, expected 2.")
        super().__init__(priors_on_parameters=priors_on_parameters, treatment_policy=treatment_policy)
        self.noise_scale = noise_scale

    @staticmethod
    def _get_mean(
        parameters: Parameters,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the mean of the output distribution, i.e. sample:

        𝑓(𝑎, 𝑐;𝛙) = tanh(𝑐 (𝛙₀𝑎)) . (𝛙₁𝑎)
        """
        layer_1 = torch.nn.Tanh()(context[..., None, None] * parameters["psi_0"])
        layer_2 = torch.sum(layer_1 * parameters["psi_1"], dim=-1)
        return layer_2

    def sample_observational_data(
        self,
        parameters: Parameters,
        context: torch.Tensor,
        effect_variable: Optional[str] = "y",
        return_mean: bool = False,
    ) -> torch.Tensor:
        """
        Defines a pyro model to sample sample from the observational distribution 𝑝(𝑦 | 𝑐, 𝑎, 𝛙).

        Args:
            parameters: dict of the form {parameter_name: torch.tensor(value)}--samples of 𝛙 from the current prior.
            context: a one-dimensional context variable.
            treatment: a one-dimensional treatment.
            return_mean: whether to return the mean of the distribution or the sampled outcome; defaults to False
                (i.e. return the observation).

        Returns:
            A sample from 𝑝(𝑦 | 𝑐, 𝑎, 𝛙) or its mean.
        """
        mean = torch.sum(self._get_mean(parameters, context) * self.treatment_policy(), dim=-1)
        y = pyro.sample(effect_variable, dist.Normal(mean, self.noise_scale).to_event(1))
        return mean if return_mean else y

    def calculate_conditional_max_reward(
        self, parameters: Parameters, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The maximum reward is defined as the maximum expected 𝑦 over all possible treatments, i.e.
            𝑚(𝑎 | 𝛙) = max_𝑎 E[𝑦 | 𝑐, do(𝑎), 𝛙].
        Since we are dealing with a fixed graph and binary 𝑎 we have 𝑚(𝑎 | 𝛙) = max (E [𝑦 | 𝑐, 0, 𝛙], E [𝑦 | 𝑐, 1, 𝛙])

        Args:
            parameters: dict of the form {parameter_name: torch.tensor(value)}--samples of 𝛙 from the current prior.
            context: context variable

        Returns: A tuple (max_reward, optimal_treatment) -- <max_reward> is the maximum reward achievable by
            <optimal_treatment>  for the given <context>
        """
        maxes = torch.max(self._get_mean(parameters, context), dim=-1)

        return maxes.values, maxes.indices
