from typing import Dict, Tuple

import pyro
import pyro.distributions as dist
import torch

from .single_confounding_root import Parameters, SingleConfoundingRoot


class ContinuousContextDiscreteTreatment(SingleConfoundingRoot):
    """
    This is a single confounding root Bayesian toy data simulator of the form ๐ฆ = ๐(๐, ๐) + ๐N(0, 1) where
    ๐ = {0, 1}^K is discrete treatment with K category represented as one hot vector , set by the user, ๐ is a 1d context, which is given,
    ๐ is a random function, parameterised by ๐โ ~ N((ฮผโโ, ฮผโโ)แต, ฮฃโ), ๐=1,...,K  as follows:

       ๐(๐, ๐; ๐) = (-๐ยฒ + ฮฒ(๐)๐ + ฮณ(๐)) . (๐)
          ฮณ(๐, ๐=๐) = (๐โโ +  ๐โโ + 18) / 2
          ฮฒ(๐, ๐=๐) = (๐โโ - ฮณ(๐, ๐โ) + 9) / 3

    Note:
        the notation ๐=๐ means that the k-th component of the treatment is 1.
        We don't place priors on ๐ and ๐. Although in the fully Bayesian single confounding root scenario
        ๐ = ๐(๐) + ๐, in the experimental design context we always choose ๐, i.e. we do(๐), and so learning
        a distribution for ๐ is not necessary.
    """

    def __init__(
        self,
        treatment_policy: torch.nn.Module,
        priors_on_parameters: Dict,
        noise_scale: float = 0.1,
        treatment_obs_pairs: bool = False,
    ) -> None:
        """
        Args:
            priors_on_parameters: priors on the parameters.
                It should be in the form {parameter_name: pyro.distributions}.
            noise_scale: Standard deviation (scale) of the observation noise, ๐
        """
        super().__init__(priors_on_parameters=priors_on_parameters, treatment_policy=treatment_policy)
        self.noise_scale = noise_scale
        self.treatment_obs_pairs = treatment_obs_pairs

    def _get_means(
        self,
        parameters: Parameters,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the means of the output distribution for the both treatments:
        """
        gamma = (parameters["c_neg"] + parameters["c_pos"] + 2 * 3.0**2) / 2.0  # shape = batch_size, 2
        beta = (parameters["c_pos"] - gamma + 3.0**2) / 3.0  # shape = batch_size, 2
        return -1.0 * context.pow(2) + beta * context + gamma

    def sample_observational_data(
        self,
        parameters: Parameters,
        context: torch.Tensor,
        effect_variable: str = "y",
        return_mean: bool = False,
    ) -> torch.Tensor:
        """
        Defines a pyro model to sample from the observational distribution ๐(๐ฆ | ๐, ๐, ๐).

        Args:
            parameters: dict of the form {parameter_name: torch.tensor(value)}--samples of ๐ from the current prior.
            context: a one-dimensional context variable.
            treatment: a one-dimensional treatment.
            return_mean: whether to return the mean of the distribution or the sampled outcome; defaults to False
                (i.e. return the observation).

        Returns:
            A sample from ๐(๐ฆ | ๐, ๐, ๐) or its mean.
        """
        treatment = self.treatment_policy()
        mean = (self._get_means(parameters, context) * treatment).sum(dim=-1)
        y = pyro.sample(effect_variable, dist.Normal(mean, self.noise_scale).to_event(1))

        if return_mean:
            return mean
        else:
            return (treatment.max(-1).indices, y) if self.treatment_obs_pairs else y

    def calculate_conditional_max_reward(
        self, parameters: Parameters, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The maximum reward is defined as the maximum expected ๐ฆ over all possible treatments, i.e.
            ๐(๐ | ๐) = max_๐ E[๐ฆ | ๐, do(๐), ๐].
        Since we are dealing with a fixed graph and binary ๐ we have ๐(๐ | ๐) = max (E [๐ฆ | ๐, 0, ๐], E [๐ฆ | ๐, 1, ๐])

        Args:
            parameters: dict of the form {parameter_name: torch.tensor(value)}--samples of ๐ from the current prior.
            context: context variable

        Returns: A tuple (max_reward, optimal_treatment) -- <max_reward> is the maximum reward achievable by
            <optimal_treatment>  for the given <context>
        """
        maxes = torch.max(self._get_means(parameters, context), dim=-1)
        return maxes.values, maxes.indices
