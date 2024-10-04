import torch
import torch.nn.functional as F

from dwma.models.classifier import GaussianNoiseActionClassifier
from dwma.models.video_diffusion_pytorch.diffusion import GaussianDiffusion


class ClassifierGuidedDiffusion(GaussianDiffusion):
    """A class for performing inference using classifier-guidance on a diffusion model.

    Follows the reference code from the OpenAI repo on guided diffusion. Specifically, see this file:
    https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py
    """

    def __init__(
        self,
        base_model: GaussianDiffusion,
        classifier: GaussianNoiseActionClassifier,
        act_cond_wt: float,
        no_act_cond_steps=0,
    ):
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
        if not hasattr(base_model.model, "cond_drop_prob"):
            print("Warning: Base model does not have cond_drop_prob attribute. Setting to 0.0.")
            base_model.model.cond_drop_prob = 0.0
        self.classifier = classifier
        self.act_cond_wt = act_cond_wt
        self.no_act_cond_steps = no_act_cond_steps

    def get_act_cond_wts(self, t):
        """Get the scale of the classifier guidance to use at each timestep."""
        weights = torch.ones_like(t) * self.act_cond_wt

        # set weights to zero when t is less than no_act_cond_steps
        weights[t < self.no_act_cond_steps] = 0.0
        bs = t.shape[0]
        weights = weights.reshape(bs, 1, 1, 1, 1)
        return weights

    @torch.no_grad()
    def p_sample(self, x, t: int, cond_frames=None, act=None, cond_drop_prob=None):
        """Modify the p sample step in the base GaussianDiffusion class to include classifier guidance."""
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            model_mean, variance, model_log_variance, x_start, _ = self.p_mean_variance(
                x=x,
                t=batched_times,
                cond_frames=cond_frames,
                act=act,
                clip_denoised=True,
                cond_drop_prob=cond_drop_prob,
            )

        # add classifier guidance
        model_mean = self.condition_mean(
            mean=model_mean, variance=variance, x=x, act=act, cond_frames=cond_frames, t=batched_times
        )

        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start, {}

    def condition_mean(self, mean, variance, x, act, cond_frames, t):
        """Compute the mean for the previous step, given that classifier guidance is applied."""
        assert act is not None, "Need actions for classifier guidance"

        classifier_scale = self.get_act_cond_wts(t)
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            action_logits = self.classifier.model_predictions(x=x_in, x_init=cond_frames, t=t)
            log_probs = F.log_softmax(action_logits, dim=-1)
            real_act_prob = log_probs.gather(2, act.unsqueeze(-1)).squeeze(-1)
            grad = torch.autograd.grad(real_act_prob.sum(), x_in)[0] * classifier_scale

        new_mean = mean.float() + variance * grad.float()
        return new_mean
