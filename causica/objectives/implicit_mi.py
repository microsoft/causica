import math
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple

import numpy as np
import pyro
import torch
from pyro.infer.util import torch_item
from pyro.util import warn_if_nan
from torch import nn
from tqdm import trange


class ImplicitMutualInformation(ABC):
    """
    In experimental design we want to maximise the mutual information between an outcome 𝑦 and
    some quantity we wish to gain information about, denoted by 𝜃, under a design 𝑎.
    Mathematically this is

    I(𝜃; 𝑦 | 𝑎) = KL(p(𝑦, 𝜃 | 𝑎) || p(𝑦 | 𝑎)p(𝜃))
                = E_{p(𝑦, 𝜃 | 𝑎)} [log(p(𝑦, 𝜃 | 𝑎)) - log(p(𝜃)p(𝑦 | 𝑎))]
                = E_{p(𝜃) p(𝑦 | 𝜃, 𝑎)} [log(p(𝑦 | 𝜃, 𝑎)) - log(p(𝑦 | 𝑎))],

    where p(y | 𝑎) = E_{p(𝜃)}[p(y | 𝜃, 𝑎)], which is usually unavailable in closed form.
    (It is typically assumed that the distribution of the quantity of interest is independent of the
    design, i.e. p(𝜃| 𝑎) = p(𝜃).)

    There are a number of ways tro approximate I(𝜃; 𝑦 | 𝑎). Here we focus on methods that rely on
    samples only, i.e. that do not require the likelihood p(𝑦 | 𝜃, 𝑎) being analytic.
    """

    def __init__(self, model, critic: nn.Module, batch_size: int) -> None:
        """
        Args:
            model: a model to produce joint samples.
            critic: a function taking 𝑦, 𝜃 and returning a number.
            batch_size: number of samples to approximate the expectation.
        """
        self.model = model
        self.critic = critic
        self.batch_size = batch_size

    @abstractmethod
    def differentiable_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def loss(self, *args, **kwargs) -> float:
        raise NotImplementedError()

    def evaluate_loss(self, *args, **kwargs) -> float:
        """
        Evaluate the loss function.
        """
        with torch.no_grad():
            loss = self.loss(*args, **kwargs)
            return torch_item(loss)

    def evaluate_mi(self, *args, n_reps: int = 10, **kwargs) -> Tuple[float, float]:
        """
        Return an estimate of the mutual information.
        Args:
            n_reps: number of samples of size batch_size to draw.
        Returns:
            Estimated mean and standard error.
        """
        res = [-self.evaluate_loss(*args, **kwargs) for _ in range(n_reps)]

        return float(np.mean(res)), float(np.std(res)) / math.sqrt(n_reps)

    def train_step(self, optim: pyro.optim.PyroOptim, *args, **kwargs):
        with pyro.poutine.trace(param_only=True) as param_capture:  # pylint: disable=not-callable
            loss = self.differentiable_loss(*args, **kwargs)
            loss.backward()

        params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())
        optim(params)
        pyro.infer.util.zero_grads(params)
        return torch_item(loss)


class InfoNCE(ImplicitMutualInformation):
    """
    InfoNCE is one possible estimator for MI that relies on joint samples only.
    """

    def __init__(self, model: nn.Module, critic: nn.Module, batch_size: int) -> None:
        super().__init__(model, critic, batch_size)
        self.num_negative_samples = batch_size - 1  # reuse samples in the denom calculation

    def _vectorize_joint_sampling(self, sizes: Iterable, name: str):
        """A utility to vectorize sampling"""

        def wrapped_fn(*args, **kwargs):
            with pyro.plate_stack(name, sizes):
                return self.model.forward(*args, **kwargs)

        return wrapped_fn

    def differentiable_loss(self, *args, **kwargs) -> torch.Tensor:
        # at each gradient step expose parameters so pyro optim recognises them
        if hasattr(self.critic, "parameters"):
            pyro.module("critic_net", self.critic)

        vectorized = self._vectorize_joint_sampling([self.batch_size], "expand_batch")
        x, y = vectorized(*args, **kwargs)

        joint_scores, product_scores = self.critic(x, y)
        joint_term = joint_scores.sum() / self.batch_size
        product_term = (joint_scores + product_scores).logsumexp(dim=1).mean()

        loss = product_term - joint_term
        warn_if_nan(loss, "loss")
        return loss

    def loss(self, *args, **kwargs) -> float:
        loss_to_constant = torch_item(self.differentiable_loss(*args, **kwargs))
        return loss_to_constant - math.log(self.num_negative_samples + 1)

    def train(
        self,
        optim: pyro.optim.PyroOptim,
        num_training_steps: int,
        *args,
        annealing_frequency: Optional[int] = None,
        **kwargs,
    ):

        num_steps_range = trange(1, num_training_steps + 1, desc="Loss: 0.000 ")
        scheduler_step = getattr(optim, "step", False)
        for i in num_steps_range:
            loss = self.train_step(optim, *args, **kwargs)
            num_steps_range.set_description(f"Loss: {loss:.3f} ")
            if scheduler_step and i % annealing_frequency == 0:
                optim.step()

    @property
    def optimal_design(self):
        try:
            return self.model.treatment_policy().detach()
        except torch.nn.modules.module.ModuleAttributeError as exc:
            raise ValueError("Design not found.") from exc
