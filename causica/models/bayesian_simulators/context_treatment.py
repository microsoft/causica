from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import pyro
import torch
from joblib import Parallel, delayed
from pyro.distributions import Empirical
from pyro.infer import MCMC, NUTS, EmpiricalMarginal, Importance

Parameters = Dict[str, torch.Tensor]


class BayesianContextTreatmentModel(ABC, torch.nn.Module):
    """
    Generic model simulator (as opposed to a real-world, environment simulator) class for Bayesian experimental
    design to learn about max reward values. We denote the Bayesian parameters by 𝛙 on which we place a prior p(𝛙).
    The parameters 𝛙 consists of a graph 𝐺 and/or parameters 𝜃. We use the following notation
        𝛙 = (𝐺, 𝜃) are the <parameters>
        𝑦 = <effect_variable>
        𝑎 = <treatment>
        𝑐 = <context>
    """

    def __init__(
        self, priors_on_parameters: Dict, treatment_policy: torch.nn.Module, return_max_reward: bool = True
    ) -> None:
        """
        Args:
            priors_on_parameters: prior on 𝛙, which should be a dict of the form {latent_name: prior_distribution}
        """
        super().__init__()
        self.priors_on_parameters = priors_on_parameters
        self._initial_priors_on_parameters = priors_on_parameters
        self.treatment_policy = treatment_policy
        self.return_max_reward = return_max_reward

    def reset_prior(self, prior: Optional[dict] = None) -> None:
        """
        Reset the prior to the original or some other prior.
        Args:
            prior: new prior (e.g. a posterior). If None (default), set the prior to the original one, i.e. the
                used at instantiation of the class.
        """
        if prior is None:
            prior = self._initial_priors_on_parameters
        self.priors_on_parameters = prior

    def sample_parameters(self) -> Parameters:
        """Return a sample of the parameters from the prior, 𝛙 ~ <priors_on_parameters>"""
        return {site_name: pyro.sample(site_name, prior) for site_name, prior in self.priors_on_parameters.items()}

    @abstractmethod
    def sample_observational_data(self, parameters: Parameters, context: torch.Tensor, effect_variable: str):
        """
        Return a sample of a <effect_variable> conditional on parameters, context and treatment, 𝑦 ~ p(𝑦 | 𝑐, 𝑎, 𝛙)
        """
        raise NotImplementedError()

    @abstractmethod
    def sample_interventional_data(self, parameters: Parameters, context: torch.Tensor, effect_variable: str):
        """
        Return a sample from 𝑦 ~ p(𝑦 | 𝑐, do(𝑎), 𝛙)
        """
        # in DECI this corresponds to .sample/.cate, but cate is expensive in general
        raise NotImplementedError()

    @abstractmethod
    def sample_counterfactual_data(
        self,
        parameters: Parameters,
        context: torch.Tensor,
        effect_variable: str,
        factual_outcome: str,
    ):
        raise NotImplementedError()

    @abstractmethod
    def calculate_conditional_max_reward(self, parameters: Parameters, context: torch.Tensor):
        """
        Calculate the max achievable expected reward for a given context 𝑐 and realisation of 𝛙, i.e.
              𝑚(𝛙, c) = max_𝑎 E [𝑦 | 𝑐, do(𝑎), 𝛙]
        returning 𝑚(𝛙, c) and the optimal treatment 𝑎 that achieves it.
        """
        raise NotImplementedError()

    def sample_joint(
        self, context_obs: torch.Tensor, context_test: Optional[torch.Tensor] = None, effect_variable: str = "y"
    ) -> Tuple[Dict[str, torch.Tensor], Any, Any]:
        """
        Sample from the joint
            p(𝛙)p(y, m | c_obs, a_obs, c_test, 𝛙) if test context_test is specified
            p(𝛙)p(y | c_obs, a_obs, 𝛙) if context_test is None,
        where
            𝑚(𝛙, c) = max_𝑎 E [𝑦 | 𝑐, do(𝑎), 𝛙]
        is the max reward defined in the <calculate_conditional_max_reward> method.
        Args:
            context_obs: observation context.
            treatment_obs: observation treatment.
            context_test: test context.
        Returns:
            A tuple (parameters, observations, max_reward) if context_test is given or (parameters, observations) otherwise.
        """
        parameters = self.sample_parameters()
        observations = self.sample_observational_data(
            parameters=parameters, context=context_obs, effect_variable=effect_variable
        )
        if context_test is not None:
            max_rewards, optimal_treatment = self.calculate_conditional_max_reward(
                parameters=parameters, context=context_test
            )
            if self.return_max_reward:
                return parameters, observations, max_rewards
            else:
                return parameters, observations, optimal_treatment

        return parameters, observations, None

    def forward(self, context_obs: torch.Tensor, context_test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward method must return the joint samples from the distributions which will be used to optimise
        mutual information.
        """

        pyro.module("treatment_policy", self.treatment_policy)

        _, observations, max_rewards = self.sample_joint(context_obs=context_obs, context_test=context_test)
        return observations, max_rewards

    def infer_model_snis(self, observed_data: Dict[str, torch.Tensor], num_samples: int):
        model_conditioned = pyro.condition(self.sample_joint, data=observed_data)
        importance = Importance(model_conditioned, guide=None, num_samples=num_samples)
        importance_run = importance.run(*observed_data["args"])
        self.priors_on_parameters = {
            site: EmpiricalMarginal(importance_run, sites=site) for site in self.priors_on_parameters.keys()
        }

    def infer_model_hmc(self, observed_data: Dict[str, torch.Tensor], num_samples: int, num_chains: int) -> None:
        """
        Condition on observational data and obtain a posterior of the parameters 𝛙 using HMC.
        This posterior is set as a prior. To reset to original prior use the <reset_prior> method.
        Args:
            observed_data: data dictionary of the form {"site_name": site_value} to be passed to pyro.codnition.
            num_samples: number of posterior HMC samples to generate. Warm-up steps is set to 0.2*num_samples.
            num_chains: number of HMC chains to run in parallel.
        """
        model_conditioned = pyro.condition(self.sample_joint, data=observed_data)
        kernel = NUTS(model_conditioned, target_accept_prob=0.6)
        num_threads = min(num_chains, 6)

        def get_samples():
            mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=int(0.2 * num_samples), num_chains=1)
            mcmc.run(*observed_data["args"])
            return mcmc.get_samples()

        all_samples = Parallel(n_jobs=num_threads)(delayed(get_samples)() for _ in range(num_chains))
        posterior_samples = {key: torch.cat([sample[key] for sample in all_samples]) for key in all_samples[0].keys()}
        self.priors_on_parameters = {
            latent: Empirical(posterior_samples[latent], log_weights=torch.ones(posterior_samples[latent].shape[0]))
            for latent in posterior_samples.keys()
        }

    def infer_model(self, inference: str, *args, **kwargs):
        if inference == "snis":
            self.infer_model_snis(*args, **kwargs)
        elif inference == "hmc":
            self.infer_model_hmc(*args, **kwargs)
        else:
            raise ValueError(f"Unknown inference={inference}")

    def _vectorize(self, fn: Callable, sizes: Iterable, name: str):
        """A utility to vectorize sampling"""

        def wrapped_fn(*args, **kwargs):
            with pyro.plate_stack(name, sizes):
                return fn(*args, **kwargs)

        return wrapped_fn
