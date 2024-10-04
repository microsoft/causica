from enum import Enum

import torch
from einops import rearrange
from lvdm.models.ddpm3d import DDPM


class CFGTypeEnum(Enum):
    PoE = "PoE"
    ActionCFG = "ActionCFG"


class MixedDiffusionWrapper:
    def __init__(
        self,
        base_model: DDPM,
        action_conditioned_model: DDPM,
        mixing_strategy: str,
        act_cond_wt: float,
        no_act_cond_steps: int = 0,
        max_steps: int = 1000,
        linear_decay: bool = False,
        use_base_ema: bool = False,
    ):
        super().__init__()
        self.base_model = base_model
        self.action_conditioned_model = action_conditioned_model

        assert mixing_strategy in CFGTypeEnum.__members__, f"Invalid mixing strategy {mixing_strategy}"
        self.mixing_strategy = CFGTypeEnum[mixing_strategy]
        self.act_cond_wt = act_cond_wt
        self.no_act_cond_steps = no_act_cond_steps
        self.max_steps = max_steps
        self.linear_decay = linear_decay
        self.use_base_ema = use_base_ema

        # use the EMA parameters from the base model
        if self.use_base_ema:
            self.base_model.model_ema.store(self.base_model.model.parameters())
            self.base_model.model_ema.copy_to(self.base_model.model)
            print("Using EMA parameters for base model.")

    def get_act_cond_wts(self, t):
        """Get the action conditioned weights for each diffusion timestep."""
        weights = torch.ones_like(t) * self.act_cond_wt

        if self.linear_decay:
            weights = weights * (t / self.max_steps)

        # set weights to zero when t is less than no_act_cond_steps
        weights[t < self.no_act_cond_steps] = 0.0
        return weights

    def __getattr__(self, name):
        if name in self.__dict__:
            return super().__getattr__(name)
        return getattr(self.base_model, name)

    def model_predictions(self, x_noisy, t, cond, **kwargs):
        """Override the model predictions method to combine predictions from base and action conditioned models."""

        act_cond_wts = rearrange(self.get_act_cond_wts(t), "b -> b 1 1 1 1")
        with torch.no_grad():

            # model predictions using base model
            basepred_noise, _, _ = self.base_model.model_predictions(x_noisy, t, cond, **kwargs)

            # model predictions using action conditioned model without action dropout
            actpred_noise, _, _ = self.action_conditioned_model.model_predictions(
                x_noisy, t, cond, dropout_actions=False, **kwargs
            )

            if self.mixing_strategy == CFGTypeEnum.PoE:
                # combine prediction
                mixed_noise_pred = basepred_noise * (1 - act_cond_wts) + actpred_noise * act_cond_wts
                mixed_x0_pred = self.base_model.predict_start_from_noise(x_noisy, t, mixed_noise_pred)
                mixed_v_pred = self.get_v(mixed_x0_pred, mixed_noise_pred, t)
                return mixed_noise_pred, mixed_v_pred, mixed_x0_pred

            # remove action conditioning and call action model again
            c_act_model_copy = cond.copy()
            c_act_model_copy["act"] = None
            actpred_noise_noact, _, _ = self.action_conditioned_model.model_predictions(
                x_noisy, t, c_act_model_copy, dropout_actions=False, **kwargs
            )

            if self.mixing_strategy == CFGTypeEnum.ActionCFG:
                mixed_noise_pred = basepred_noise + act_cond_wts * (actpred_noise - actpred_noise_noact)
                mixed_x0_pred = self.base_model.predict_start_from_noise(x_noisy, t, mixed_noise_pred)
                mixed_v_pred = self.get_v(mixed_x0_pred, mixed_noise_pred, t)
                return mixed_noise_pred, mixed_v_pred, mixed_x0_pred
            raise ValueError(f"Invalid mixing strategy {self.cfg_type}")
