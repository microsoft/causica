from typing import Dict, Optional, Tuple

import pyro
import pyro.distributions as dist
import torch

from .single_confounding_root import Parameters, SingleConfoundingRoot


class ContinuousContextAndTreatment(SingleConfoundingRoot):
    """
    This is a single confounding root Bayesian toy data simulator of the form 𝑦 = 𝑓(𝑎, 𝑐) + 𝜎N(0, 1) where
    𝑎 is a 1d treatment, set by the user, 𝑐 is a 1d context, which is given, and 𝑓 is a random function,
    parameterised by 𝛙 = (𝜓₀, 𝜓₁, 𝜓₂, 𝜓₃) as follows:

       𝑓(𝑎, 𝑐; 𝛙) = exp  (- (𝑎 - g(𝛙, c))² / h(𝛙, c)²- λ𝑎²), λ >= 0 is a constant <cost_weight>

    The function 𝑔 determines the location of the maximum expected 𝑦 and depends on the parameters 𝛙 and the
    context 𝑐. Specifically 𝑔(𝛙, 𝑐) =  𝜓₀ + 𝜓₁*𝑐 + 𝜓₂*𝑐² and the maximum expected 𝑦 is 1, which is
    achieved by setting 𝑎 to 𝑔(𝛙, 𝑐).

    The function 𝘩 controls the scale of 𝑓 and equals 𝘩(𝛙, 𝑐) = 𝜓₃, i.e. it is independent if the context 𝑐.

    Note:
        We don't place priors on 𝑎 and 𝑐. Although in the fully Bayesian single confounding root scenario
        𝑎 = 𝑓(𝑐) + 𝜀, in the experimental design context we always choose 𝑎, i.e. we do(𝑎), and so learning
        a distribution for 𝑎 is not necessary.
    """

    def __init__(
        self,
        treatment_policy: torch.nn.Module,
        priors_on_parameters: Dict,
        noise_scale: float = 0.1,
        cost_weight: float = 0.1,
        return_max_reward: bool = True,
    ) -> None:
        """
        Args:
            priors_on_parameters: dict of length 4 containing the priors (pyro.distributions) on the 4 parameters.
                It should be in the form {parameter_name: pyro.distributions}. The parameters 𝛙  are required to be positive.
            noise_scale: Standard deviation (scale) of the noise, 𝜎
            cost_weight: penalise large treatments, defaults to 0.1
        """
        super().__init__(
            priors_on_parameters=priors_on_parameters,
            treatment_policy=treatment_policy,
            return_max_reward=return_max_reward,
        )
        self.noise_scale = noise_scale
        self.cost_weight = cost_weight

    def _costless_optimal_treatment(self, parameters: Parameters, context: torch.Tensor) -> torch.Tensor:
        """
        Calculate 𝑔(𝛙, 𝑐) =  𝜓₀ + 𝜓₁*𝑐 + 𝜓₂*𝑐² which is also the optimal treatment if λ=0
        Args:
            parameters: sample of size <batch_size> from the (current) prior; each psi{i} is expected to be of shape [batch_size_psi, 1]
                context: 1-d context variable, expected shape [1, batch_size_context] to allow broadcasting.
        Returns:
            A tensor of shape [batch_size_psi, batch_size_context]
        """
        return parameters["psi0"] + parameters["psi1"] * context - parameters["psi2"] * context**2

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
        costless_max_location = self._costless_optimal_treatment(parameters, context)
        mean = torch.exp(
            -(((self.treatment_policy() - costless_max_location) / parameters["psi3"]) ** 2)
            - self.cost_weight * self.treatment_policy() ** 2
        )
        y = pyro.sample(effect_variable, dist.Normal(mean, self.noise_scale).to_event(1))
        return mean if return_mean else y

    def calculate_conditional_max_reward(
        self, parameters: Parameters, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The maximum reward is defined as the maximum expected 𝑦 oval all possible treatments, i.e.
            𝑚(𝑎 | 𝛙) = max_𝑎 E[𝑦 | 𝑐, do(𝑎), 𝛙].
        Since we are dealing with a fixed graph we gave 𝑚(𝑎 | 𝛙) = max_𝑎 E [𝑦 | 𝑐, 𝑎, 𝛙].

        Args:
            parameters: dict of the form {parameter_name: torch.tensor(value)}--samples of 𝛙 from the current prior.
            context: context variable

        Returns: A tuple (max_reward, optimal_treatment) -- <max_reward> is the maximum reward achievable by
            <optimal_treatment>  for the given <context>

        Details on obtaining max reward:
            Note max_𝑎 E [𝑦 | 𝑐, 𝑎, 𝛙] <=> max_𝑎 𝑓(𝑎, 𝑐). The derivative with respect to 𝑎 of 𝑓(𝑎, 𝑐) is
                𝑓' = -2𝑓(𝑎, 𝑐)  ( λ𝑎h(𝛙, c)² + 𝑎 - g(𝛙, c) ) / h(𝛙, c)²
            Setting this to 0 gives us the optimal treatment:
                𝑎* = g(𝛙, c) / (1 + λ*h(𝛙, c)^2)
            and substituting back in 𝑓 gives the max reward.
        """

        costless_max_location = self._costless_optimal_treatment(parameters, context)
        optimal_treatment = costless_max_location / (1 + self.cost_weight * parameters["psi3"] ** 2)
        max_reward = torch.exp(
            -(
                (
                    self.cost_weight
                    * parameters["psi3"]
                    * costless_max_location
                    / (1 + self.cost_weight * parameters["psi3"] ** 2)
                )
                ** 2
            )
            - self.cost_weight * optimal_treatment**2
        )
        return max_reward, optimal_treatment
