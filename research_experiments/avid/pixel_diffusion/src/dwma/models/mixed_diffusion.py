from enum import Enum

import torch

from dwma.models.video_diffusion_pytorch.diffusion import GaussianDiffusion, ModelPrediction


class CFGTypeEnum(Enum):
    PoE = "PoE"
    BaseCFG = "BaseCFG"
    ActionCFG = "ActionCFG"


class MixedGaussianDiffusion(GaussianDiffusion):
    """A class to combine predictions from a base diffusion model and an action-conditioned diffusion model."""

    def __init__(
        self,
        base_model: GaussianDiffusion,
        act_cond_model: GaussianDiffusion,
        act_cond_wt: float,
        cfg_type: str,
        no_act_cond_steps=0,
    ):
        """
        Args:
            base_model: the base video-only model that will not receive action-conditioning
            act_cond_model: the action-conditioned model that will receive action-conditioning
            act_cond_wt: the weight to apply to the action-conditioned model
            no_act_cond_steps: the number of steps at the beginning of the diffusion process to not apply
                action-conditioning
        """
        if not hasattr(base_model.model, "cond_drop_prob"):
            print("Warning: Base model does not have cond_drop_prob attribute. Setting to 0.0.")
            base_model.model.cond_drop_prob = 0.0
        if not hasattr(act_cond_model.model, "cond_drop_prob"):
            print("Warning: Action conditioned model does not have cond_drop_prob attribute. Setting to 0.0.")
            act_cond_model.model.cond_drop_prob = 0.0

        super().__init__(
            model=base_model.model,
            image_size=base_model.image_size,
            timesteps=base_model.timesteps,
            sampling_timesteps=base_model.sampling_timesteps,
            objective=base_model.objective,
            beta_schedule=base_model.beta_schedule,
            schedule_fn_kwargs=base_model.schedule_fn_kwargs,
            ddim_sampling_eta=base_model.ddim_sampling_eta,
            auto_normalize=base_model.auto_normalize,
            offset_noise_strength=base_model.offset_noise_strength,
            min_snr_loss_weight=base_model.min_snr_loss_weight,
            min_snr_gamma=base_model.min_snr_gamma,
        )
        self.act_cond_model = act_cond_model
        self.act_cond_wt = act_cond_wt
        self.no_act_cond_steps = no_act_cond_steps
        self.cfg_type = getattr(CFGTypeEnum, cfg_type)

    def get_act_cond_wts(self, t):
        """Get the action conditioned weights for each diffusion timestep."""
        weights = torch.ones_like(t) * self.act_cond_wt

        # set weights to zero when t is less than no_act_cond_steps
        weights[t < self.no_act_cond_steps] = 0.0
        bs = t.shape[0]
        weights = weights.reshape(bs, 1, 1, 1, 1)
        return weights

    def model_predictions(
        self, x, t, cond_frames=None, act=None, cond_drop_prob=None, clip_x_start=False, rederive_pred_noise=False
    ):
        """Override the model predictions method to combine predictions from base and action conditioned models."""
        if act is None:
            print("Warning: No actions provided to mixed prediction. Neither model will be action-conditioned.")

        with torch.no_grad():
            # model predictions using base model
            base_preds, _ = super().model_predictions(
                x=x,
                t=t,
                cond_frames=cond_frames,
                act=None,
                cond_drop_prob=0.0,
                clip_x_start=clip_x_start,
                rederive_pred_noise=rederive_pred_noise,
            )

            # model predictions using action conditioned model
            act_cond_preds, _ = self.act_cond_model.model_predictions(
                x=x,
                t=t,
                cond_frames=cond_frames,
                act=act,
                cond_drop_prob=0.0,
                clip_x_start=clip_x_start,
                rederive_pred_noise=rederive_pred_noise,
            )

            if self.cfg_type == CFGTypeEnum.PoE:

                # combine prediction
                act_cond_wts = self.get_act_cond_wts(t)
                mixed_noise_pred = base_preds.pred_noise * (1 - act_cond_wts) + act_cond_preds.pred_noise * act_cond_wts
                mixed_x_start_pred = self.predict_start_from_noise(x, t, mixed_noise_pred)
                return ModelPrediction(mixed_noise_pred, mixed_x_start_pred), {}

            act_cond_preds_no_cond, _ = self.act_cond_model.model_predictions(
                x=x,
                t=t,
                cond_frames=cond_frames,
                act=act,
                cond_drop_prob=1.0,
                clip_x_start=clip_x_start,
                rederive_pred_noise=rederive_pred_noise,
            )

            if self.cfg_type == CFGTypeEnum.BaseCFG:
                # combine prediction
                act_cond_wts = self.get_act_cond_wts(t)
                mixed_noise_pred = base_preds.pred_noise + act_cond_wts * (
                    act_cond_preds.pred_noise - base_preds.pred_noise
                )
                mixed_x_start_pred = self.predict_start_from_noise(x, t, mixed_noise_pred)
                return ModelPrediction(mixed_noise_pred, mixed_x_start_pred), {}
            if self.cfg_type == CFGTypeEnum.ActionCFG:
                # combine prediction
                act_cond_wts = self.get_act_cond_wts(t)
                mixed_noise_pred = base_preds.pred_noise + act_cond_wts * (
                    act_cond_preds.pred_noise - act_cond_preds_no_cond.pred_noise
                )
                mixed_x_start_pred = self.predict_start_from_noise(x, t, mixed_noise_pred)
                return ModelPrediction(mixed_noise_pred, mixed_x_start_pred), {}
            raise ValueError("Invalid CFGTypeEnum")
