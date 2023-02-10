from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np
import torch
from dataclasses_json import dataclass_json
from torch.optim import Optimizer


class AugLagLossCalculator:
    def __init__(self, init_alpha: float, init_rho: float):
        self.alpha = init_alpha
        self.rho = init_rho

    def __call__(self, objective: torch.Tensor, constraint: torch.Tensor) -> torch.Tensor:
        return objective + self.alpha * constraint + self.rho * constraint * constraint / 2


@dataclass_json
@dataclass(frozen=True)
class AugLagLRConfig:
    """
    Configuration parameters for the AuglagLR scheduler.

    lr_update_lag (int): Number of iterations to wait before updating the learning rate.
    lr_update_lag_best (int): Number of iterations to wait after the best model before updating the learning rate.
    lr_init_dict (Dict[str, float]): Dictionary of intitialization parameters for every new inner optimization step.
        This must contain all parameter_groups for all optimizers
    aggregation_period (int): Aggregation period to compare the mean of the loss terms across this period.
    lr_factor (float): Learning rate update schedule factor (exponential decay).
    penalty_progress_rate (float): Number of iterations to wait before updating rho based on the dag penalty.
    safety_rho (float): Maximum rho that could be updated to.
    safety_alpha (float): Maximum alpha that could be udated to.
    max_lr_down (int): Maximum number of lr update times to decide inner loop termination.
    inner_early_stopping_patience (int): Maximum number of iterations to run after the best inner loss to terminate inner loop.
    max_outer_steps (int): Maximum number of outer update steps.
    patience_penalty_reached (int): Maximum number of outer iterations to run after the dag penalty has reached a good value.
    patience_max_rho (int): Maximum number of iterations to run once rho threshold is reached.
    penalty_tolerance (float): Tolerance of the dag penalty
    max_inner_steps (int): Maximum number of inner loop steps to run.

    """

    lr_update_lag: int = 500
    lr_update_lag_best: int = 250
    lr_init_dict: Dict[str, float] = field(
        default_factory=lambda: {"vardist": 0.1, "icgnn": 0.0003, "noise_dist": 0.003}
    )
    aggregation_period: int = 20
    lr_factor: float = 0.1
    penalty_progress_rate: float = 0.65
    safety_rho: float = 1e13
    safety_alpha: float = 1e13
    max_lr_down: int = 3
    inner_early_stopping_patience: int = 500
    max_outer_steps: int = 100
    patience_penalty_reached: int = 5
    patience_max_rho: int = 3
    penalty_tolerance: float = 1e-5
    max_inner_steps: int = 3000


class AugLagLR:
    def __init__(self, config: AugLagLRConfig) -> None:
        """A Pytorch like scheduler which performs the Augmented Lagrangian optimization procedure, which consists of
        an inner loop which optimizes the objective for a fixed set of lagrangian parameters. The lagrangian parameters are
        annealed in the outer loop, according to a schedule as specified by the hyperparameters.

        Args:
            config: An `AugLagLRConfig` object containing the configuration parameters.
        """
        self.config = config

        self.outer_opt_counter = 0
        self.outer_below_penalty_tol = 0
        self.outer_max_rho = 0
        self._prev_lagrangian_penalty = np.inf
        self._cur_lagrangian_penalty = np.inf

        self.best_loss = np.inf
        self.last_lr_update_step = 0
        self.num_lr_updates = 0
        self.last_best_step = 0
        self.loss_tracker: deque = deque([], maxlen=config.aggregation_period)
        self.step_counter = 0
        self.epoch_counter = 0

    def _init_new_inner_optimisation(self):
        """Init the hyperparameters for a new inner loop optimization."""
        self.best_loss = np.inf
        self.last_lr_update_step = 0
        self.num_lr_updates = 0
        self.last_best_step = 0
        self.loss_tracker.clear()
        self.step_counter = 0
        self.epoch_counter = 0

    def _is_inner_converged(self) -> bool:
        """Check if the inner optimization loop has converged, based on maximum number of inner steps, number of lr updates.

        Returns:
            bool: Return True if converged, else False.
        """
        return (
            self.step_counter >= self.config.max_inner_steps
            or self.num_lr_updates >= self.config.max_lr_down
            or self.last_best_step + self.config.inner_early_stopping_patience <= self.step_counter
        )

    def _is_outer_converged(self) -> bool:
        """Check if the outer loop has converged based on the main outer opt counter, if the dag penalty is below a threshold
        or if the rho parameter has reached a limit"

        Returns:
            bool: Return True if converged, else False
        """
        return (
            self.outer_opt_counter >= self.config.max_outer_steps
            or self.outer_below_penalty_tol >= self.config.patience_penalty_reached
            or self.outer_max_rho >= self.config.patience_max_rho
        )

    def _enough_steps_since_last_lr_update(self) -> bool:
        """Check if enough steps have been taken since the previous learning rate update, based on the previous
        update step iteration.

        Returns:
            bool: indicating whether sufficient steps have occurred since the last update
        """
        return self.last_lr_update_step + self.config.lr_update_lag <= self.step_counter

    def _enough_steps_since_best_model(self) -> bool:
        """Check the number of iteration steps which have been passed after seeing the current best model.

        Returns:
            bool: Returns True if last iteration at which learning rate was
            updated and last best loss iteration is less than total steps, else False.
        """
        return self.last_best_step + self.config.lr_update_lag_best <= self.step_counter

    def _update_lr(self, optimizer: Union[Optimizer, List[Optimizer]]):
        """Update the learning rate of the optimizer(s) based on the lr multiplicative factor.

        Args:
            optimizer (Union[Optimizer, List[Optimizer]]): Optimizers of auglag to be updated.
        """
        self.last_lr_update_step = self.step_counter
        self.num_lr_updates += 1

        if isinstance(optimizer, list):
            for opt in optimizer:
                for param_group in opt.param_groups:
                    param_group["lr"] *= self.config.lr_factor
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= self.config.lr_factor

    def _reset_lr(self, optimizer: Union[Optimizer, List[Optimizer]]):
        """Reset the learning rate of individual param groups from lr init dictionary.

        Args:
            optimizer (Union[Optimizer, List[Optimizer]]): Optimizer(s) corresponding to all param groups.
        """
        self.last_lr_update_step = self.step_counter

        if isinstance(optimizer, list):
            for opt in optimizer:
                for param_group in opt.param_groups:
                    param_group["lr"] = self.config.lr_init_dict[param_group["name"]]

        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.config.lr_init_dict[param_group["name"]]

    def _update_lagrangian_params(self, loss: AugLagLossCalculator):
        """Update the lagrangian parameters (of the auglag routine) based on the dag constraint values observed.

        Args:
            loss (AugLagLoss): loss with lagrangian attributes rho and alpha to be updated.
        """
        if self._cur_dag_penalty < self.config.penalty_tolerance:
            self.outer_below_penalty_tol += 1
        else:
            self.outer_below_penalty_tol = 0

        if loss.rho > self.config.safety_rho:
            self.outer_max_rho += 1

        if self._cur_dag_penalty > self._prev_lagrangian_penalty * self.config.penalty_progress_rate:
            print(f"Updating rho, dag penalty prev: {self._prev_lagrangian_penalty: .10f}")
            loss.rho *= 10.0
        else:
            self._prev_lagrangian_penalty = self._cur_dag_penalty
            loss.alpha += loss.rho * self._cur_dag_penalty
            if self._cur_dag_penalty == 0.0:
                loss.alpha *= 5
            print(f"Updating alpha to: {loss.alpha}")
        if loss.rho >= self.config.safety_rho:
            loss.alpha *= 5

        loss.rho = min([loss.rho, self.config.safety_rho])
        loss.alpha = min([loss.alpha, self.config.safety_alpha])

    def _is_auglag_converged(self, optimizer: Union[Optimizer, List[Optimizer]], loss: AugLagLossCalculator) -> bool:
        """Checks if the inner and outer loops have converged. If inner loop is converged,
        it initilaizes the optimisation parameters for a new inner loop. If both are converged, it returns True.

        Args:
            optimizer (Union[Optimizer, List[Optimizer]]): Optimizer(s) corresponding to different parameter groups on which auglag is being performed.
            loss (AugLagLoss): Auglag loss.

        Returns:
            bool: Returns True if both inner and outer have converged, else False
        """
        if self._is_inner_converged():
            if self._is_outer_converged():
                return True
            else:
                self._update_lagrangian_params(loss)
                self.outer_opt_counter += 1
                self._init_new_inner_optimisation()
                self._reset_lr(optimizer)
        elif self._enough_steps_since_last_lr_update() and self._enough_steps_since_best_model():
            self._update_lr(optimizer)

        return False

    def _check_best_loss(self):
        """Update the best loss based on the average loss over an aggregation period."""
        if len(self.loss_tracker) == self.config.aggregation_period:
            avg_loss = np.mean(self.loss_tracker)
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.last_best_step = self.step_counter

    def step(
        self,
        optimizer: Union[Optimizer, List[Optimizer]],
        loss: AugLagLossCalculator,
        loss_value: float,
        lagrangian_penalty: float,
    ) -> bool:
        """The main update method to take one auglag inner step.

        Args:
            optimizer (Union[Optimizer, List[Optimizer]]): Optimizer(s) corresponding to different param groups.
            loss (AugLagLoss): auglag loss with lagrangian parameters
            loss_value (float): the actual value of the elbo for the current update step.
            lagrangian_penalty (float): Dag penalty for the current update step.

        Returns:
            bool: if the auglag has converged (False) or not (True)
        """
        assert lagrangian_penalty >= 0, "auglag penalty must be non-negative"
        self.loss_tracker.append(loss_value)
        self._cur_dag_penalty = lagrangian_penalty
        self.step_counter += 1
        self._check_best_loss()
        return self._is_auglag_converged(optimizer=optimizer, loss=loss)
