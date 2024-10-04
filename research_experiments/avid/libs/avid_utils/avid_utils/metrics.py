import numpy as np
import pytorch_lightning as pl
import torch
from avid_utils.image import preprocess_images
from avid_utils.video import resize_video
from cdfvd import fvd
from einops import rearrange
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def update_running_mean(old_samples, old_mean, new_samples, new_mean):
    """Update the running mean.

    Args:
        old_samples: The number of samples in the old mean
        old_mean: The old mean
        new_samples: The number of new samples
        new_mean: The mean of the new samples

    Returns:
        The updated mean
    """
    numerator = old_samples * old_mean + new_samples * new_mean
    denominator = old_samples + new_samples
    return numerator / denominator


class Metric:
    """Base class for all metrics."""

    def reset_state(self):
        """Reset the state of the metric."""
        raise NotImplementedError

    def update(
        self,
        cond_frames: torch.Tensor,
        pred_frames: torch.Tensor,
        real_frames: torch.Tensor,
        real_act: torch.Tensor | None = None,
    ):
        """Update the estimate of the metric with new data.

        Args:
            cond_frames: The conditioning frames (B, T, C, H, W) in [0, 255]
            pred_frames: The predicted frames (B, T, C, H, W) in [0, 255]
            real_frames: The real frames (B, T, C, H, W) in [0, 255]
        """
        raise NotImplementedError

    def log_dict(self, prefix: str = "") -> dict:
        """Return a dictionary of the metric values."""
        raise NotImplementedError

    def check_valid(self, real_frames, pred_frames):
        assert (
            real_frames.dtype == torch.uint8 and pred_frames.dtype == torch.uint8
        ), "Expected videos to be of type uint8"
        assert real_frames.shape == pred_frames.shape, "Expected real and fake videos to have the same shape"
        assert real_frames.ndim == 5, "Expected videos to be of shape (b, t, h, w, c)"
        assert real_frames.shape[-1] == 3, "Expected videos to have 3 channels"


class MSE(Metric):
    """Mean squared error metric."""

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.mse_per_step = {}
        self.frame_count = {}

    def update(
        self,
        cond_frames: torch.Tensor,
        pred_frames: torch.Tensor,
        real_frames: torch.Tensor,
        real_act: torch.Tensor | None = None,
    ):
        self.check_valid(real_frames, pred_frames)
        new_samples = pred_frames.shape[0]
        errs = preprocess_images(pred_frames) - preprocess_images(real_frames)  # (b, c, t, h, w)
        mse_per_step = torch.mean(errs**2, dim=(0, 1, 3, 4))  # average across b, c, h, w

        # update stats at each step length
        for step in range(mse_per_step.shape[0]):
            old_mean = self.mse_per_step.get(step, 0)
            old_samples = self.frame_count.get(step, 0)
            new_mean = mse_per_step[step].item()
            self.mse_per_step[step] = update_running_mean(old_samples, old_mean, new_samples, new_mean)
            self.frame_count[step] = old_samples + new_samples

    def log_dict(self, prefix: str = "") -> dict:
        metric_dict = {f"{prefix}/metrics/mse_step_{step}": val for step, val in self.mse_per_step.items()}
        metric_dict[f"{prefix}/metrics/mse"] = sum(self.mse_per_step.values()) / len(self.mse_per_step.values())
        return metric_dict


class FVD(Metric):
    """Fréchet video distance metric."""

    def __init__(self, num_video_frames: int, resize_size=(224, 224), add_cond_frames=True):
        if num_video_frames < 10:
            print("Using videomae for FVD as less than 10 frames.")
            self.fvd_evaluator = fvd.cdfvd("videomae")
            self.fvd_type = "videomae"
        else:
            self.fvd_evaluator = fvd.cdfvd("i3d")
            self.fvd_type = "i3d"
        self.resize_size = resize_size
        self.fvd_score = None
        self.add_cond_frames = add_cond_frames
        self.reset_state()

    def reset_state(self):
        self.fvd_evaluator.empty_fake_stats()
        self.fvd_evaluator.empty_real_stats()

    def update(
        self,
        cond_frames: torch.Tensor,
        pred_frames: torch.Tensor,
        real_frames: torch.Tensor,
        real_act: torch.Tensor | None = None,
    ):
        self.check_valid(real_frames, pred_frames)

        # for fvd we will eval entire video including conditioning frames
        if self.add_cond_frames:
            real_frames = torch.cat([cond_frames, real_frames], dim=1)
            pred_frames = torch.cat([cond_frames, pred_frames], dim=1)

        real_frames = resize_video(real_frames, self.resize_size)
        pred_frames = resize_video(pred_frames, self.resize_size)
        with torch.no_grad():
            fvd_score = self.fvd_evaluator.compute_fvd(
                real_videos=real_frames.cpu().numpy(), fake_videos=pred_frames.cpu().numpy()
            )
        self.fvd_score = fvd_score

    def log_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}/metrics/fvd_{self.fvd_type}": self.fvd_score}


class SSIM(Metric):
    """Structural similarity index metric."""

    def __init__(self, device):
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.reset_state()

    def reset_state(self):
        self.ssim_per_step = {}
        self.frame_count = {}

    def update(
        self,
        cond_frames: torch.Tensor,
        pred_frames: torch.Tensor,
        real_frames: torch.Tensor,
        real_act: torch.Tensor | None = None,
    ):
        self.check_valid(real_frames, pred_frames)
        new_samples = pred_frames.shape[0]

        for t in range(real_frames.shape[1]):
            real_frames_t = rearrange(real_frames[:, t], "b h w c -> b c h w") / 255.0
            pred_frames_t = rearrange(pred_frames[:, t], "b h w c -> b c h w") / 255.0
            # pylint: disable=not-callable
            ssim_t = self.ssim(real_frames_t, pred_frames_t)

            old_ssim = self.ssim_per_step.get(t, 0)
            old_samples = self.frame_count.get(t, 0)
            self.ssim_per_step[t] = update_running_mean(old_samples, old_ssim, new_samples, ssim_t)
            self.frame_count[t] = old_samples + new_samples

    def log_dict(self, prefix: str = "") -> dict:
        metric_dict = {f"{prefix}/metrics/ssim_step_{step}": val for step, val in self.ssim_per_step.items()}
        metric_dict[f"{prefix}/metrics/ssim_avg"] = sum(self.ssim_per_step.values()) / len(self.ssim_per_step.values())
        return metric_dict


class LPIPS(Metric):
    """Learned perceptual image patch similarity metric."""

    def __init__(self, device, net_type="vgg"):
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type).to(device)
        self.reset_state()

    def reset_state(self):
        self.lpips_per_step = {}
        self.frame_count = {}

    def update(
        self,
        cond_frames: torch.Tensor,
        pred_frames: torch.Tensor,
        real_frames: torch.Tensor,
        real_act: torch.Tensor | None = None,
    ):
        self.check_valid(real_frames, pred_frames)
        new_samples = pred_frames.shape[0]

        # lpips expects input in range [-1, 1]
        pred_frames = preprocess_images(pred_frames)
        real_frames = preprocess_images(real_frames)  # (b, c, t, h, w)

        for t in range(real_frames.shape[2]):
            real_frames_t = real_frames[:, :, t]
            pred_frames_t = pred_frames[:, :, t]
            # pylint: disable=not-callable
            lpips_t = self.lpips(real_frames_t, pred_frames_t)

            old_lpips = self.lpips_per_step.get(t, 0)
            old_samples = self.frame_count.get(t, 0)
            self.lpips_per_step[t] = update_running_mean(old_samples, old_lpips, new_samples, lpips_t)
            self.frame_count[t] = old_samples + new_samples

    def log_dict(self, prefix: str = "") -> dict:
        metric_dict = {f"{prefix}/metrics/lpips_step_{step}": val for step, val in self.lpips_per_step.items()}
        metric_dict[f"{prefix}/metrics/lpips_avg"] = sum(self.lpips_per_step.values()) / len(
            self.lpips_per_step.values()
        )
        return metric_dict


class PSNR(Metric):
    """PSNR metric."""

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.psnr_per_step = {}
        self.frame_count = {}

    def update(
        self,
        cond_frames: torch.Tensor,
        pred_frames: torch.Tensor,
        real_frames: torch.Tensor,
        real_act: torch.Tensor | None = None,
    ):
        self.check_valid(real_frames, pred_frames)
        new_samples = pred_frames.shape[0]
        errs = (preprocess_images(pred_frames) + 1.0) / 2.0 - (
            preprocess_images(real_frames) + 1.0
        ) / 2.0  # norm to [0,1], calculate errors (b, c, t, h, w)
        mse_values = torch.mean(errs**2, dim=(1, 3, 4))  # average across c, h, w to give (b, t)
        mse_values = torch.clamp(mse_values, min=1e-10)

        psnr_values = 10 * torch.log10(1 / mse_values)  # (b, t)
        psnr_values = psnr_values.mean(dim=0)  # average across b to give (t)

        # update stats at each step length
        for step in range(psnr_values.shape[0]):
            old_mean = self.psnr_per_step.get(step, 0)
            old_samples = self.frame_count.get(step, 0)
            new_mean = psnr_values[step].item()
            self.psnr_per_step[step] = update_running_mean(old_samples, old_mean, new_samples, new_mean)
            self.frame_count[step] = old_samples + new_samples

    def log_dict(self, prefix: str = "") -> dict:
        metric_dict = {f"{prefix}/metrics/psnr_step_{step}": val for step, val in self.psnr_per_step.items()}
        metric_dict[f"{prefix}/metrics/psnr"] = sum(self.psnr_per_step.values()) / len(self.psnr_per_step.values())
        return metric_dict


class ActionAccuracy(Metric):
    """Metric to evaluate how accurately the actions can be predicted from the generated videos and real videos.

    Args:
        classifier_module: An accurate action classifier to be used for the evaluation
    """

    def __init__(self, classifier_module: pl.LightningModule):
        self.classifier_module = classifier_module
        self.reset_state()

    def reset_state(self):
        self.acc_real_frames = 0
        self.acc_gen_frames = 0
        self.act_prob_real_frames = 0
        self.act_prob_gen_frames = 0
        self.err_real_frames = 0
        self.err_gen_frames = 0
        self.sample_count = 0

    def update(
        self,
        cond_frames: torch.Tensor,
        pred_frames: torch.Tensor,
        real_frames: torch.Tensor,
        real_act: torch.Tensor | None = None,
    ):
        self.check_valid(real_frames, pred_frames)
        assert real_act is not None, "Expected real actions to be provided for action accuracy metric"

        new_samples = pred_frames.shape[0]
        if hasattr(self.classifier_module, "get_prediction_accuracy"):
            with torch.no_grad():
                acc_real_frames, act_prob_real_frames = self.classifier_module.get_prediction_accuracy(
                    cond_frames, real_frames, real_act
                )
                acc_gen_frames, act_prob_gen_frames = self.classifier_module.get_prediction_accuracy(
                    cond_frames, pred_frames, real_act
                )
            self.acc_real_frames = update_running_mean(
                self.sample_count, self.acc_real_frames, new_samples, acc_real_frames
            )
            self.acc_gen_frames = update_running_mean(
                self.sample_count, self.acc_gen_frames, new_samples, acc_gen_frames
            )
            self.act_prob_real_frames = update_running_mean(
                self.sample_count, self.act_prob_real_frames, new_samples, act_prob_real_frames
            )
            self.act_prob_gen_frames = update_running_mean(
                self.sample_count, self.act_prob_gen_frames, new_samples, act_prob_gen_frames
            )

        if hasattr(self.classifier_module, "get_prediction_error"):
            with torch.no_grad():
                err_real_frames = self.classifier_module.get_prediction_error(cond_frames, real_frames, real_act)
                err_gen_frames = self.classifier_module.get_prediction_error(cond_frames, pred_frames, real_act)
            self.err_real_frames = update_running_mean(
                self.sample_count, self.err_real_frames, new_samples, err_real_frames
            )
            self.err_gen_frames = update_running_mean(
                self.sample_count, self.err_gen_frames, new_samples, err_gen_frames
            )
        self.sample_count += new_samples

    def log_dict(self, prefix: str = "") -> dict:
        metric_dict = {}
        if hasattr(self.classifier_module, "get_prediction_accuracy"):
            metric_dict.update(
                {
                    f"{prefix}/metrics/act_acc_real_frames": self.acc_real_frames,
                    f"{prefix}/metrics/act_acc_gen_frames": self.acc_gen_frames,
                    f"{prefix}/metrics/act_acc_ratio": (self.acc_gen_frames / (self.acc_real_frames + 1e-9)),
                    f"{prefix}/metrics/act_act_prob_real_frames": self.act_prob_real_frames,
                    f"{prefix}/metrics/act_act_prob_gen_frames": self.act_prob_gen_frames,
                    f"{prefix}/metrics/act_act_prob_ratio": (
                        self.act_prob_gen_frames / (self.act_prob_real_frames + 1e-9)
                    ),
                }
            )
        if hasattr(self.classifier_module, "get_prediction_error"):
            metric_dict.update(
                {
                    f"{prefix}/metrics/act_err_real_frames": self.err_real_frames,
                    f"{prefix}/metrics/act_err_gen_frames": self.err_gen_frames,
                    f"{prefix}/metrics/act_err_ratio": (self.err_gen_frames / (self.err_real_frames + 1e-9)),
                }
            )
        return metric_dict


class FID(Metric):
    """Adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py"""

    def __init__(self, device, batch_size=128, dims=2048):
        self.reset_state()
        self.batch_size = batch_size
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.model = InceptionV3([block_idx]).to(device)
        self.device = device

    def reset_state(self):
        self.real_stats = []
        self.fake_stats = []

    def update(
        self,
        cond_frames: torch.Tensor,
        pred_frames: torch.Tensor,
        real_frames: torch.Tensor,
        real_act: torch.Tensor | None = None,
    ):
        self.check_valid(real_frames, pred_frames)
        real_frames = resize_video(real_frames, (299, 299))
        pred_frames = resize_video(pred_frames, (299, 299))
        real_frames = rearrange(real_frames, "b t h w c -> (b t) c h w") / 255.0
        pred_frames = rearrange(pred_frames, "b t h w c -> (b t) c h w") / 255.0

        # loop over batches
        for i in range(0, real_frames.shape[0], self.batch_size):
            real_batch = real_frames[i : i + self.batch_size].to(self.device)
            pred_batch = pred_frames[i : i + self.batch_size].to(self.device)

            with torch.no_grad():
                output_real = self.model(real_batch)[0]  # pylint: disable=E1102
                output_fake = self.model(pred_batch)[0]  # pylint: disable=E1102
            output_real = output_real.squeeze(3).squeeze(2).cpu().numpy()
            output_fake = output_fake.squeeze(3).squeeze(2).cpu().numpy()
            self.real_stats.append(output_real)
            self.fake_stats.append(output_fake)

    def log_dict(self, prefix: str = "") -> dict:
        metric_dict = {}
        if len(self.real_stats) == 0 or len(self.fake_stats) == 0:
            return metric_dict

        # concatenate all the stats along batch dimension
        real_stats_all = np.concatenate(self.real_stats, axis=0)
        fake_stats_all = np.concatenate(self.fake_stats, axis=0)

        mu_real = np.mean(real_stats_all, axis=0)
        mu_fake = np.mean(fake_stats_all, axis=0)

        sigma_real = np.cov(real_stats_all, rowvar=False)
        sigma_fake = np.cov(fake_stats_all, rowvar=False)

        fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        metric_dict[f"{prefix}/metrics/fid"] = fid_value

        return metric_dict
