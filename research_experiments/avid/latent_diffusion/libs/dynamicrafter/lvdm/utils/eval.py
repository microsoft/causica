from contextlib import nullcontext

import torch
import wandb
from lvdm.utils.save_video import prepare_to_log
from lvdm.utils.utils import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger
from torch.cuda.amp import autocast


def get_run_logger(config, run_kwargs=None):
    if run_kwargs is None:
        run_kwargs = {}

    # add the special kwargs to the config
    for key, value in run_kwargs.items():
        setattr(config, key, value)

    # add the special kwargs to the run name
    run_name = config.name + "_" + "_".join([f"{k}={v}" for k, v in run_kwargs.items()])

    # convert config to dict
    config_dict = OmegaConf.to_container(config, resolve=True)

    return WandbLogger(
        name=run_name,
        group=config.group,
        project=config.logger.params.project,
        entity=config.logger.params.entity,
        offline=config.logger.params.offline,
        save_dir=config.logger.params.save_dir,
        config=config_dict,
    )


# Function to move tensors to the device
def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    else:
        return batch


def evaluate_and_log(
    eval_config,
    diffusion_model,
    device,
    sampler,
    dataloader,
    num_batches,
    run_kwargs=None,
    fixed_action=None,
    precision=16,
):
    # set logger for these inference params
    run_logger = get_run_logger(eval_config, run_kwargs=run_kwargs)
    video_logger = instantiate_from_config(eval_config.video_logger_callback)
    video_logger.set_logger(run_logger)
    dataloader_iter = iter(dataloader)

    for i in range(num_batches):
        batch = next(dataloader_iter)
        val_batch = move_to_device(batch, device)

        # base model and action model may have different conditioning signal due to different embedding funcs
        z, c, uc, cond_mask, logs, kwargs = diffusion_model.prepare_batch_for_inference(val_batch)

        if fixed_action is not None:
            print("Running inference with fixed action: ", fixed_action)
            fixed_action_tensor = torch.tensor(fixed_action, device=device)
            fixed_action_tensor = fixed_action_tensor.reshape(1, 1, -1)  # (1, 1, action_dim)
            fixed_action_tensor = fixed_action_tensor.repeat(
                c["act"].shape[0], c["act"].shape[1], 1
            )  # (b, t, action_dim)
            c["act"] = fixed_action_tensor

        # prepare args for ddim
        ddim_kwargs = OmegaConf.to_container(eval_config.ddim_kwargs, resolve=True)
        ddim_kwargs.update(kwargs)
        shape = (
            diffusion_model.channels,
            diffusion_model.temporal_length,
            *diffusion_model.image_size,
        )

        print("DDIM kwargs: ", ddim_kwargs)

        amp_context = autocast() if precision == 16 else nullcontext()
        print("Using precision: ", precision, " for inference")

        # sample from diffusion model and decode
        diffusion_model.eval()
        with diffusion_model.ema_scope("Plotting"):
            with amp_context:
                samples, _ = sampler.sample(
                    ddim_kwargs["ddim_steps"],
                    batch_size=z.shape[0],
                    shape=shape,
                    conditioning=c,
                    unconditional_conditioning=uc,
                    mask=cond_mask,
                    x0=z,
                    **ddim_kwargs,
                )
            x_samples = diffusion_model.decode_first_stage(samples)
        logs["samples"] = x_samples
        actions = c["act"]

        # log the videos and metrics
        video_logger.log_metrics(diffusion_model, logs, actions, split="val", batch_idx=i)
        logs = prepare_to_log(logs, video_logger.max_images, video_logger.clamp)
        torch.cuda.empty_cache()
        video_logger.log_media(diffusion_model, logs, split="val")

    # close the wandb run
    wandb.finish()
