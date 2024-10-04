import math
from collections import namedtuple
from functools import partial

import torch
import torch.nn.functional as F
from avid_utils.helpers import default, extract
from avid_utils.normalization import identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from einops import rearrange, reduce
from torch import nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from dwma.models.video_diffusion_pytorch.unet3d import Unet3D

ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def get_beta_schedule_fn(beta_schedule):
    if beta_schedule == "linear":
        beta_schedule_fn = linear_beta_schedule
    elif beta_schedule == "cosine":
        beta_schedule_fn = cosine_beta_schedule
    elif beta_schedule == "sigmoid":
        beta_schedule_fn = sigmoid_beta_schedule
    else:
        raise ValueError(f"unknown beta schedule {beta_schedule}")
    return beta_schedule_fn


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: Unet3D,
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
        super().__init__()
        assert not (isinstance(self, GaussianDiffusion) and model.channels != model.out_dim)
        assert not hasattr(model, "random_or_learned_sinusoidal_cond") or not model.random_or_learned_sinusoidal_cond
        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v",
        }, "objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v \
            (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in \
            imagen-video successfully])"

        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.objective = objective
        self.beta_schedule = beta_schedule
        self.schedule_fn_kwargs = schedule_fn_kwargs
        self.ddim_sampling_eta = ddim_sampling_eta
        self.auto_normalize = auto_normalize
        self.offset_noise_strength = offset_noise_strength
        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

        self.channels = self.model.channels

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        assert (
            isinstance(image_size, (tuple, list)) and len(image_size) == 2
        ), "image size must be a integer or a tuple/list of two integers"
        self.image_size = image_size

        beta_schedule_fn = get_beta_schedule_fn(beta_schedule)
        if schedule_fn_kwargs is None:
            schedule_fn_kwargs = {}
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # derive loss weight
        # snr - signal noise ratio

        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == "pred_noise":
            self.register_buffer("loss_weight", maybe_clipped_snr / snr)
        elif objective == "pred_x0":
            self.register_buffer("loss_weight", maybe_clipped_snr)
        elif objective == "pred_v":
            self.register_buffer("loss_weight", maybe_clipped_snr / (snr + 1))

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def register_buffer(self, name, val):
        return super().register_buffer(name, val.to(torch.float32))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

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
        model_output = self.model(x, t, cond_frames=cond_frames, act=act, cond_drop_prob=cond_drop_prob)

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

        info = {}
        return ModelPrediction(pred_noise, x_start), info

    def p_mean_variance(self, x, t, cond_frames=None, act=None, cond_drop_prob=None, clip_denoised=True):
        preds, info = self.model_predictions(x, t, cond_frames=cond_frames, act=act, cond_drop_prob=cond_drop_prob)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start, info

    @torch.no_grad()
    def p_sample(self, x, t: int, cond_frames=None, act=None, cond_drop_prob=None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start, info = self.p_mean_variance(
            x=x, t=batched_times, cond_frames=cond_frames, act=act, clip_denoised=True, cond_drop_prob=cond_drop_prob
        )
        noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start, info

    @torch.no_grad()
    def p_sample_loop(self, shape, return_all_timesteps=False, cond_frames=None, act=None, cond_drop_prob=None):
        _, device = shape[0], self.device

        img = torch.randn(shape, device=device)
        imgs = [img]
        infos = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps):
            img, _, info = self.p_sample(img, t, cond_frames=cond_frames, act=act, cond_drop_prob=cond_drop_prob)
            imgs.append(img)  #
            infos.append(info)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)
        info = {key: [d[key] for d in infos] for key in infos[0]}  # convert to dict of lists

        ret = self.unnormalize(ret)
        return ret, info

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps=False, cond_frames=None, act=None, cond_drop_prob=None):
        batch, device, total_timesteps, sampling_timesteps, eta, _ = (
            shape[0],
            self.device,
            self.num_timesteps,
            self.sampling_timesteps,
            self.ddim_sampling_eta,
            self.objective,
        )

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img,
                time_cond,
                cond_frames=cond_frames,
                act=act,
                cond_drop_prob=cond_drop_prob,
                clip_x_start=True,
                rederive_pred_noise=True,
            )

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim=1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, batch_size=16, return_all_timesteps=False, cond_frames=None, act=None, cond_drop_prob=None):
        (h, w), channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        if hasattr(self.model, "generation_frames"):
            shape = (batch_size, channels, self.model.generation_frames, h, w)
        else:
            shape = (batch_size, channels, h, w)
        return sample_fn(
            shape,
            return_all_timesteps=return_all_timesteps,
            cond_frames=cond_frames,
            act=act,
            cond_drop_prob=cond_drop_prob,
        )

    @autocast(enabled=False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

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
        model_out = self.model(x, t, cond_frames=cond_frames, act=act)

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

    def forward(self, img, *args, **kwargs):
        b = img.shape[0]
        h = img.shape[-2]
        w = img.shape[-1]
        device = img.device

        assert (
            h == self.image_size[0] and w == self.image_size[1]
        ), f"height and width of image must be {self.image_size}"
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        loss, _, mask = self.p_losses(img, t, *args, **kwargs)
        info = {}
        if mask is not None:
            info["mask_mean"] = mask.mean().item()
            info["mask_std"] = mask.std().item()
            info["mask_min"] = mask.min().item()
            info["mask_max"] = mask.max().item()
        return loss, info
