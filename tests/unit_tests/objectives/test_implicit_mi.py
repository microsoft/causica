import math
from typing import Callable

import pyro
import pyro.distributions as dist
import pytest
import torch
from torch import nn

from causica.models.bayesian_simulators.context_treatment import BayesianContextTreatmentModel, Parameters
from causica.models.bayesian_simulators.treatment_policy import IdentityTreatmentPolicy
from causica.objectives.implicit_mi import InfoNCE


class GsnModel(BayesianContextTreatmentModel):
    """
    Simulate two correlated Gaussians with correlation=<target_correlation> in the <Simulator> API
    """

    def __init__(self, target_correlation: float):
        priors_on_parameters = {"x": dist.Normal(torch.tensor([0.0]), torch.tensor([1.0])).to_event(1)}
        super().__init__(
            priors_on_parameters, treatment_policy=IdentityTreatmentPolicy(torch.tensor([0.0]), learnable=False)
        )
        self.target_correlation = target_correlation
        self.beta = target_correlation / math.sqrt(1 - target_correlation**2)
        self.true_mi = -0.5 * math.log(1 - target_correlation**2)

    def sample_parameters(self) -> Parameters:
        return {key: pyro.sample(key, value) for key, value in self.priors_on_parameters.items()}

    def sample_observational_data(self, parameters: Parameters, *_):
        """Simple pyro model that generates two correlated Gaussians"""
        # mark one of the variables as parameter sample
        y = pyro.sample("y", dist.Normal(self.beta * parameters["x"], 1.0).to_event(1))
        return y

    def forward(self, **_):
        parameters = self.sample_parameters()
        obs = self.sample_observational_data(parameters, None, None)
        #  InfoNCE expects sample_joint to return <parameters, obs, max_rewards>
        return (self.treatment_policy.treatment, obs), parameters

    def sample_joint(self):
        pass

    def infer_model(self):
        pass

    def calculate_conditional_max_reward(self):
        pass

    def sample_counterfactual_data(self):
        pass

    def sample_interventional_data(self):
        pass


class GsnSourceFinding(BayesianContextTreatmentModel):
    """
    α ~ N(z, 1)
    y = 1 / (0.1 + (α - design)^2) + N(0, 1.0)

    Optimal design is equals z
    """

    def __init__(self, treatment: torch.Tensor, parameter_mean: float, noise_scale: float):
        priors_on_parameters = {"alpha": dist.Normal(torch.tensor([parameter_mean]), torch.tensor([0.1])).to_event(1)}
        super().__init__(priors_on_parameters, treatment_policy=IdentityTreatmentPolicy(treatment))
        self.parameter_mean = parameter_mean
        self._noise_scale = noise_scale

    def sample_parameters(self):
        return {key: pyro.sample(key, value) for key, value in self.priors_on_parameters.items()}

    def sample_observational_data(self, parameters: Parameters, *_):
        """Simple pyro model"""
        # mark one of the variables as parameter sample
        mean = 1 / (0.01 + (parameters["alpha"] - self.treatment_policy.forward()) ** 2)
        y = pyro.sample("y", dist.Normal(mean, self._noise_scale).to_event(1))
        return y

    def forward(self, **_):
        pyro.param("treatment", self.treatment_policy.treatment)
        parameters = self.sample_parameters()
        obs = self.sample_observational_data(parameters, None, None)
        return (self.treatment_policy.treatment, obs), parameters

    def sample_joint(self):
        pass

    def infer_model(self):
        pass

    def calculate_conditional_max_reward(self):
        pass

    def sample_counterfactual_data(self):
        pass

    def sample_interventional_data(self):
        pass


class OptimalCritic(nn.Module):
    """
    Optimal critic for the the Gaussian models
    """

    def __init__(self, transform: Callable, noise_scale: float):
        super().__init__()
        # self.beta = beta
        self._transform = transform
        self._noise_scale = noise_scale

    def forward(self, design_obs, parameters: Parameters):
        design, obs = design_obs
        mean = self._transform(parameters, design)
        all_scores = (
            torch.distributions.Normal(mean.unsqueeze(1), self._noise_scale)
            .log_prob(obs)
            .squeeze(-1)
            .T  # need parameters in columns; observations across rows
        )
        pos_mask = torch.eye(all_scores.shape[0])
        neg_mask = 1.0 - pos_mask
        scores_joint = all_scores * pos_mask
        scores_prod = all_scores * neg_mask
        return scores_joint, scores_prod


def calculate_mi(gsn_model: GsnModel, optimal_critic: nn.Module, n_reps: int = 1000):
    """
    Estimate MI using InfoNCE estimator
    """
    # to ensure upper bound is high enough
    batch_size = int(128 * math.exp(gsn_model.true_mi))

    loss_nce = InfoNCE(gsn_model, critic=optimal_critic, batch_size=batch_size)
    mi_estimates = [loss_nce.loss(context_obs=None, context_test=None) for _ in range(n_reps)]

    return -sum(mi_estimates) / n_reps


@pytest.mark.parametrize(
    ["target_correlation", "seed"],
    [
        pytest.param(0.9, 1, id="cor_90"),
        pytest.param(0.7, 2, id="cor_70"),
        pytest.param(0.5, 3, id="cor_50"),
        pytest.param(0.3, 4, id="cor_30"),
        pytest.param(0.1, 5, id="cor_10"),
    ],
)
def test_implicit_mi(target_correlation: float, seed: int) -> None:
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    gsn_model = GsnModel(target_correlation=target_correlation)

    def transform(parameters: Parameters, *_):
        return gsn_model.beta * next(iter(parameters.values()))

    optimal_critic = OptimalCritic(transform, noise_scale=1.0)

    mi_estimate = calculate_mi(gsn_model, optimal_critic)
    # relative tolerace of 5%, absolute of 0.0001
    assert pytest.approx(mi_estimate, rel=0.05, abs=1e-4) == gsn_model.true_mi


@pytest.mark.parametrize(
    ["optimal_design_target", "seed"],
    [pytest.param(1.0, 1, id="design_target_1.0"), pytest.param(1.5, 1, id="design_target_1.5")],
)
def test_optimise_design(optimal_design_target: float, seed: int) -> None:
    pyro.set_rng_seed(seed)
    pyro.clear_param_store()

    noise_scale = 0.05
    batch_size = 128
    lr = 0.001

    def transform(parameters: Parameters, design):
        return 1 / (0.01 + (next(iter(parameters.values())) - design) ** 2)

    optimal_critic = OptimalCritic(transform=transform, noise_scale=noise_scale)

    # for test purposes choose initial 50% larger than the true one
    init_design = torch.tensor(0.1)
    gsn_source = GsnSourceFinding(treatment=init_design, parameter_mean=optimal_design_target, noise_scale=noise_scale)
    mi_instance = InfoNCE(gsn_source, critic=optimal_critic, batch_size=batch_size)

    optim = pyro.optim.Adam({"lr": lr})
    mi_instance.train(optim=optim, num_training_steps=1500, annealing_frequency=None)
    optimised_design = mi_instance.optimal_design
    # After 3K gradient steps, ask for optimal design to be within 30% of true one -- this is quite loose!
    assert pytest.approx(optimised_design.item(), rel=0.3) == optimal_design_target
