from functools import partial

import torch
import torch.nn.functional as F
from avid_utils.helpers import default, extract
from avid_utils.normalization import identity
from einops import rearrange, reduce

from dwma.models.control_net import ControlledUnetModel, ControlNet
from dwma.models.video_diffusion_pytorch.diffusion import GaussianDiffusion, ModelPrediction


class GaussianDiffusionWithControlNet(GaussianDiffusion):
    def __init__(
        self,
        model: ControlledUnetModel,
        control_net: ControlNet,
        image_size=256,
        timesteps=1000,
        sampling_timesteps=None,
        objective="pred_v",
        beta_schedule="sigmoid",
        schedule_fn_kwargs=None,
        ddim_sampling_eta=0.0,
        auto_normalize=True,
        offset_noise_strength=0.0,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight=False,  # https://arxiv.org/abs/2303.09556
        min_snr_gamma=5,
    ):
        super().__init__(
            model,
            image_size,
            timesteps,
            sampling_timesteps,
            objective,
            beta_schedule,
            schedule_fn_kwargs,
            ddim_sampling_eta,
            auto_normalize,
            offset_noise_strength,
            min_snr_loss_weight,
            min_snr_gamma,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.control_net = control_net

    def model_predictions(
        self,
        x,
        t,
        cond_frames=None,
        act=None,
        cond_drop_prob=None,
        clip_x_start=False,
        rederive_pred_noise=False,
    ):
        assert act is not None
        control_outputs = self.control_net(x, t, cond_frames=cond_frames, act=act, cond_drop_prob=cond_drop_prob)
        model_output = self.model(
            x,
            t,
            cond_frames=cond_frames,
            act=None,
            cond_drop_prob=cond_drop_prob,
            control=control_outputs,
            only_mid_control=False,
        )

        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity

        if self.objective == "pred_noise":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_x0":
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == "pred_v":
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start), {}

    def p_losses(self, x_start, t, noise=None, offset_noise_strength=None, cond_frames=None, act=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.0:
            offset_noise = torch.randn(x_start.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, "b c -> b c 1 1")

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step
        assert act is not None
        control_outputs = self.control_net(x, t, cond_frames=cond_frames, act=act)
        model_out = self.model(x, t, cond_frames=cond_frames, act=None, control=control_outputs, only_mid_control=False)

        if self.objective == "pred_noise":
            target = noise
            pred_x_start = self.predict_start_from_noise(x, t, model_out)
        elif self.objective == "pred_x0":
            target = x_start
            pred_x_start = model_out
        elif self.objective == "pred_v":
            v = self.predict_v(x_start, t, noise)
            target = v
            pred_x_start = self.predict_start_from_v(x, t, model_out)
        else:
            raise ValueError(f"unknown objective {self.objective}")

        loss = F.mse_loss(model_out, target, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean(), pred_x_start, None
