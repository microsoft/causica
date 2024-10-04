import logging
import os
import time

import wandb

mainlogger = logging.getLogger("mainlogger")

from functools import partial

import pytorch_lightning as pl
import torch
from avid_utils.image import revert_preprocess_images
from avid_utils.metrics import FID, FVD, LPIPS, MSE, PSNR, SSIM, ActionAccuracy
from einops import rearrange
from lvdm.models.action_predictor import ActionPredictor
from lvdm.utils.save_video import prepare_to_log
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        logger=None,
        max_images=1000,
        clamp=True,
        rescale=True,
        log_images_kwargs=None,
        save_fps=2,
        save_dir=None,
        reset_metrics_per_batch=True,
        action_classifier_ckpt=None,
        max_wandb_images=None,
    ):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.save_dir = save_dir
        self.clamp = clamp
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.save_fps = save_fps
        self.eval_metrics = None
        self.reset_metrics_per_batch = reset_metrics_per_batch
        self.logger = logger
        self.action_classifier_ckpt = action_classifier_ckpt
        if max_wandb_images is None:
            self.max_wandb_images = max_images
        else:
            self.max_wandb_images = max_wandb_images

    def set_logger(self, logger):
        self.logger = logger

    def prepare_eval_metrics(self, pl_module):
        self.eval_metrics = [
            MSE(),
            PSNR(),
            FVD(num_video_frames=pl_module.temporal_length, add_cond_frames=False),
            SSIM(device=pl_module.device),
            LPIPS(device=pl_module.device),
            FID(device=pl_module.device),
        ]

        if self.action_classifier_ckpt is not None:
            classifier_module = ActionPredictor.load_from_checkpoint(  # pylint: disable=E1120
                checkpoint_path=self.action_classifier_ckpt,
            )
            classifier_module.to(pl_module.device)
            self.eval_metrics.append(ActionAccuracy(classifier_module))

    def reset_metrics(self) -> None:
        """Reset the statistics of all of the eval metrics"""
        if self.eval_metrics is not None:
            for metric in self.eval_metrics:
                metric.reset_state()

    @rank_zero_only
    def log_media(self, pl_module, batch_logs, split):
        """Log images and videos to wandb.

        Args:
            pl_module: lightning module that was a WandbLogger.
            batch_logs: A dictionary of keys and values containing images and videos to be logged.
            split: The split of the data used to generate the images and videos.
        """
        logger = self.logger if self.logger else pl_module.logger
        for key in batch_logs:
            value = batch_logs[key]
            if isinstance(value, torch.Tensor) and value.dim() == 5:
                videos = revert_preprocess_images(value)
                videos = rearrange(videos, "b t h w c -> b t c h w")
                if isinstance(logger, WandbLogger):
                    for i in range(videos.shape[0]):
                        if i < self.max_wandb_images:
                            logger.experiment.log(
                                {f"media/{split}/{i}_{key}": wandb.Video(videos[i], fps=self.save_fps)},
                                commit=False,
                            )
            elif isinstance(value, torch.Tensor) and value.dim() == 4:
                imgs = revert_preprocess_images(value)
                imgs = rearrange(imgs, "b c h w -> b c h w")
                if isinstance(logger, WandbLogger):
                    for i in range(imgs.shape[0]):
                        if i < self.max_wandb_images:
                            logger.experiment.log(
                                {f"media/{split}/{i}_{key}": wandb.Image(imgs[i])},
                                commit=False,
                            )
            else:
                pass

    @rank_zero_only
    def log_metrics(self, pl_module, batch_logs, actions, split, batch_idx):
        """Log metrics to wandb.

        Args:
            pl_module: lightning module that was a WandbLogger.
            batch_logs: A dictionary of keys and values containing metrics to be logged.
            split: The split of the data used to generate the images and videos.
        """
        if self.eval_metrics is None:
            self.prepare_eval_metrics(pl_module)

        if self.reset_metrics_per_batch:
            self.reset_metrics()

        real_video = revert_preprocess_images(torch.clamp(batch_logs["real"], -1.0, 1.0))
        predicted_video = revert_preprocess_images(torch.clamp(batch_logs["samples"], -1.0, 1.0))
        conditioning = revert_preprocess_images(torch.clamp(batch_logs["image_condition"], -1.0, 1.0))

        if not self.logger:
            log_func = partial(pl_module.log_dict, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        else:
            log_func = partial(self.logger.log_metrics, step=batch_idx)

        for metric in self.eval_metrics:
            metric.update(conditioning, predicted_video, real_video, actions)
            metric_dict = metric.log_dict(split)
            log_func(metric_dict)

    @rank_zero_only
    def log_batch_imgs(self, pl_module, batch, batch_idx, split="train"):
        """Generate images, then save and log.

        pl_module: lightning module that has a log_images method.
        batch: The batch of data to be logged.
        batch_idx: The index of the batch.
        split: The split of the data.
        ddim_sampler: [Optional] The DDIMSampler to use to generate images. If not provided the pl_module will be used.
        """
        if batch_idx % self.batch_freq == 0:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            torch.cuda.empty_cache()

            for key in batch.keys():
                batch[key] = batch[key][: self.max_images]
            with torch.no_grad():
                batch_logs, actions = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
            self.log_metrics(pl_module, batch_logs, actions, split, batch_idx)

            # move to CPU and clamp
            batch_logs = prepare_to_log(batch_logs, self.max_images, self.clamp)
            torch.cuda.empty_cache()
            self.log_media(pl_module, batch_logs, split)

            if is_train:
                pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if self.batch_freq != -1 and pl_module.logdir:
            self.log_batch_imgs(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        ## different with validation_step() that saving the whole validation set and only keep the latest,
        ## it records the performance of every validation (without overwritten) by only keep a subset
        if self.batch_freq != -1 and pl_module.logdir:
            self.log_batch_imgs(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        # lightning update
        if int((pl.__version__).split(".")[1]) >= 7:
            gpu_index = trainer.strategy.root_device.index
        else:
            gpu_index = trainer.root_gpu
        torch.cuda.reset_peak_memory_stats(gpu_index)
        torch.cuda.synchronize(gpu_index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if int((pl.__version__).split(".")[1]) >= 7:
            gpu_index = trainer.strategy.root_device.index
        else:
            gpu_index = trainer.root_gpu
        torch.cuda.synchronize(gpu_index)
        max_memory = torch.cuda.max_memory_allocated(gpu_index) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass
