import pytorch_lightning as pl
import torch
import yaml
from avid_utils.image import preprocess_images, revert_preprocess_images
from avid_utils.metrics import FID, FVD, LPIPS, MSE, PSNR, SSIM, ActionAccuracy
from einops import rearrange
from ema_pytorch import EMA
from pytorch_lightning.cli import instantiate_module
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import nn
from torch.optim import Adam

from dwma.lightning.modules.classifier_module import ClassifierModule
from dwma.models.control_net import ControlNetSmaller
from dwma.models.diffusion_control_net import GaussianDiffusionWithControlNet
from dwma.models.video_diffusion_pytorch.diffusion import GaussianDiffusion


class DiffusionModule(pl.LightningModule):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion | GaussianDiffusionWithControlNet,
        learning_rate: float = 1e-4,
        condition_frames: int = 1,
        num_pred_frames: int = 1,
        sampling_frames: int = 8,
        num_samples: int = 64,
        ema_decay=0.995,
        ema_update_every=10,
        sample_every=1000,
        linear_warmup_steps=1000,
        action_cond=False,
        ckpt_path: str = None,
        action_classifier_ckpt_path: str = None,
        action_classifier_cfg_path: str = None,
    ):
        """Initialize the diffusion module.

        Args:
            diffusion_model: Diffusion model to use for training
            learning_rate: Learning rate for the optimizer
            condition_frames: Number of frames to condition on
            target_images: Number of target that the model generates (one for an image model or more than one for vid)
            sampling_frames: Number of frames to sample
            num_samples: Number of samples to generate
            ema_decay: Decay rate for the EMA
            sample_every: Log samples every n steps
            ema_update_every: Update the EMA every n steps
            action_cond: If True, condition on actions
            action_classifier_ckpt_path: Path to an accurate action classifier for evaluation purposes
            action_classifier_cfg_path: Path to the config file for the action classifier
        """
        super().__init__()
        self.diffusion_model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
        self.set_eval_model(self.ema.ema_model)
        self.learning_rate = learning_rate
        self.num_samples = num_samples
        self.condition_frames = condition_frames
        self.sample_every = sample_every
        self.action_cond = action_cond
        self.num_pred_frames = num_pred_frames
        self.linear_warmup_steps = linear_warmup_steps
        self.action_classifier_ckpt_path = action_classifier_ckpt_path
        self.action_classifier_cfg_path = action_classifier_cfg_path
        self.eval_metrics = None

        if self.num_pred_frames > 1:
            print("For video model, number of sampling frames will be equal to number of predicted frames.")
            self.sampling_frames = self.num_pred_frames
        else:
            self.sampling_frames = sampling_frames
        if ckpt_path is not None:
            self.load_statedict_from_checkpoint(ckpt_path)
        self.save_hyperparameters()

    def load_statedict_from_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["state_dict"]
        self.load_state_dict(state_dict, strict=False)
        if isinstance(self.diffusion_model, GaussianDiffusionWithControlNet):
            # Only load the model state dict if it is not a smaller control net model
            if not isinstance(self.diffusion_model.control_net, ControlNetSmaller):
                self.diffusion_model.control_net.load_state_dict(self.diffusion_model.model.state_dict(), strict=False)
            self.diffusion_model.model.eval()
            self.diffusion_model.model.requires_grad_(False)

    def set_eval_model(self, model: nn.Module) -> None:
        self.eval_model = model

    def prepare_eval_metrics(self) -> None:
        self.eval_metrics = [
            MSE(),
            FVD(num_video_frames=self.condition_frames + self.num_pred_frames),
            SSIM(device=self.device),
            LPIPS(device=self.device),
            FID(device=self.device),
            PSNR(),
        ]

        if self.action_classifier_ckpt_path is not None:
            with open(self.action_classifier_cfg_path, "r") as file:  # pylint: disable=W1514
                config = yaml.safe_load(file)

            model_config = config["model"]
            classifier_module = instantiate_module(class_type=ClassifierModule, config=model_config)
            # load the weights
            checkpoint = torch.load(self.action_classifier_ckpt_path, map_location="cpu")
            classifier_module.load_state_dict(checkpoint["state_dict"])
            classifier_module.to(self.device)
            self.eval_metrics.append(ActionAccuracy(classifier_module))

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        loss, info = self.step(batch, self.diffusion_model)
        self.log("train/loss", loss)
        for key, value in info.items():
            self.log(f"train/{key}", value)
        self.ema.update()
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval_model.eval()
        with torch.no_grad():
            loss, info = self.step(batch, self.eval_model)
        self.log("val/loss", loss)
        for key, value in info.items():
            self.log(f"val/{key}", value)

    def step(self, batch: dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
        cond_images = preprocess_images(batch["obs"][:, : self.condition_frames])
        targ_images = preprocess_images(
            batch["obs"][:, self.condition_frames : self.condition_frames + self.num_pred_frames]
        ).squeeze()

        # if using actions condition on most recent action before target frame
        if self.action_cond:

            # image model does not expect time dim in action
            if self.num_pred_frames == 1:
                act = batch["act"][:, self.condition_frames - 1]
            else:
                act = batch["act"][:, self.condition_frames - 1 : self.condition_frames + self.num_pred_frames - 1]
            loss, info = model(targ_images, cond_frames=cond_images, act=act)
        else:
            loss, info = model(targ_images, cond_frames=cond_images)
        return loss, info

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        # linear warmup multiplier
        def lr_lambda(step: int) -> float:
            return min(1.0, step / self.linear_warmup_steps)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",  # update the learning rate every step
        }
        return [optimizer], [scheduler]

    @rank_zero_only
    def on_train_batch_end(self, _, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        if batch_idx % self.sample_every == 0:
            obs = batch["obs"][: self.num_samples]
            act = batch["act"][: self.num_samples]
            self.reset_metrics()  # reset video metrics before sampling while training
            self.generate_videos(obs=obs, act=act, prefix="train", step=self.global_step, logger=self.logger)

    @rank_zero_only
    def on_validation_batch_end(self, _, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        if batch_idx % self.sample_every == 0:
            obs = batch["obs"][: self.num_samples]
            act = batch["act"][: self.num_samples]
            self.reset_metrics()
            self.generate_videos(obs=obs, act=act, prefix="val", step=self.global_step, logger=self.logger)

    def generate_videos(
        self, obs: torch.Tensor, act: torch.Tensor, prefix: str, step: int, logger: WandbLogger
    ) -> None:
        """Generate videos using the diffusion model and log them to the logger.

        Args:
            obs: Tensor of observations in shape (b, t, h, w, c) to use as conditioning
            act: Tensor of actions in shape (b, t, a) if actions for conditioning should be passed to diffusion
                sampler. If None, no actions are passed.
            prefix: Prefix to use for the logger
            step: Step to use for the logger
            logger: Logger to use for logging the videos
        """
        act_for_sample = act if self.action_cond else None
        cond, pred, real, _ = self.sample_video(real_obs=obs, real_act=act_for_sample)
        self.log_metrics(cond=cond, pred=pred, real=real, act=act, prefix=prefix, step=step, logger=logger)

        # concatenate frames with condition frames to log video
        pred = torch.cat([cond, pred], dim=1)
        real = torch.cat([cond, real], dim=1)
        vids = [real, pred]

        # add black frame to end of each vid
        for i, vid in enumerate(vids):
            vids[i] = torch.cat([vid, torch.zeros_like(vid[:, 0:1])], dim=1)

        self.log_videos(vids, prefix, step, logger)

    def log_metrics(
        self,
        cond: torch.Tensor,
        pred: torch.Tensor,
        real: torch.Tensor,
        act: torch.Tensor,
        prefix: str,
        step: int,
        logger: WandbLogger,
    ) -> None:
        """Log all of the eval metrics to the logger."""
        if self.eval_metrics is None:
            self.prepare_eval_metrics()

        for metric in self.eval_metrics:
            metric.update(cond, pred, real, real_act=act)
            metric_dict = metric.log_dict(prefix)
            logger.log_metrics(metric_dict, step=step)

    def reset_metrics(self) -> None:
        """Reset the statistics of all of the eval metrics"""
        if self.eval_metrics is not None:
            for metric in self.eval_metrics:
                metric.reset_state()

    def log_videos(self, videos: list[torch.Tensor], prefix: str, step: int, logger) -> None:
        videos = torch.cat(videos, dim=3).detach().cpu()  # conctenate the videos horizontally
        videos = rearrange(videos, "b t h w c -> b t c h w")
        if isinstance(logger, WandbLogger):
            for i in range(videos.shape[0]):
                logger.log_video(f"{prefix}/videos/{i}", [videos[i]], step=step, fps=[1])

    def log_per_step_errors(self, errors: torch.Tensor, error_name: str, prefix: str, step: int, logger) -> None:
        for i, error in enumerate(errors):
            logger.log_metrics({f"{prefix}/{error_name}_step_{i + 1}": error}, step=step)

    def sample_video(self, real_obs: torch.Tensor, real_act: torch.Tensor | None) -> torch.Tensor:
        """Sample entire video at once using video diffusion model"""
        num_samples = real_obs.shape[0]
        cond_frames = preprocess_images(real_obs[:, : self.condition_frames])
        if real_act is not None:
            act = real_act[:, self.condition_frames - 1 : self.condition_frames + self.num_pred_frames - 1]
        else:
            act = None
        generated_frames, info = self.eval_model.sample(
            batch_size=num_samples, cond_frames=cond_frames, act=act, cond_drop_prob=0.0
        )
        gen_frames_unnorm = revert_preprocess_images(generated_frames)
        cond_frames_unnorm = real_obs[:, : self.condition_frames]
        real_frames_unnorm = real_obs[:, self.condition_frames : self.condition_frames + self.num_pred_frames]
        return cond_frames_unnorm, gen_frames_unnorm, real_frames_unnorm, info
