import argparse

import torch
from lvdm.ema import LitEma
from lvdm.models.ddpm3d import DiffusionWrapper
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.utils.eval import evaluate_and_log
from lvdm.utils.train import get_model
from lvdm.utils.utils import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=0, help="seed for seed_everything")
    parser.add_argument("--config", "-e", type=str, help="path to config file")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda:0")

    eval_config = OmegaConf.load(args.config)
    base_model_config = OmegaConf.load(eval_config.base_config_file).model
    action_model_config = OmegaConf.load(eval_config.control_config_file).model

    #### Reload Action Conditioned Model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # reload action conditioned unet from appropriate checkpoint
    action_cond_unet = DiffusionWrapper.load_from_checkpoint(
        eval_config.action_model_checkpoint,
        diff_model_config=action_model_config.params.unet_config,
        conditioning_key=action_model_config.params.conditioning_key,
    )

    # reload the EMA state dict if it exists
    ckpt = torch.load(eval_config.action_model_checkpoint)
    if "ema_state_dict" in ckpt:
        ema_action_cond_unet = LitEma(action_cond_unet)
        ema_action_cond_unet.load_state_dict(ckpt["ema_state_dict"])
    else:
        ema_action_cond_unet = None

    #### Reload Base Model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    base_model_config.target = eval_config.target_module
    base_model_config.pretrained_checkpoint = eval_config.base_model_checkpoint
    controlnet_model = get_model(base_model_config)

    # Put the two models together into output adapter
    controlnet_model.prepare_adapter(action_cond_unet, ema_control_net=ema_action_cond_unet)
    controlnet_model.to(device)

    #### Prepare data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    dataset = instantiate_from_config(eval_config.data)
    dataset.setup()
    dataloader = dataset.val_dataloader()

    #### Evaluate >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    sampler = DDIMSampler(controlnet_model)
    evaluate_and_log(
        eval_config,
        controlnet_model,
        device,
        sampler,
        dataloader,
        eval_config.num_batches,
    )
