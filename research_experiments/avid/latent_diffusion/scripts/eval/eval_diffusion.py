import argparse

import torch
from lvdm.ema import LitEma
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
    model_config = OmegaConf.load(eval_config.model_config_file).model

    #### Reload Model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # instantiate the diffusion model to load VAE, etc.
    diffusion_model = get_model(model_config)

    # if using the original model we need to create EMA as it does not exist in checkpoint
    if eval_config.model_config_file == "configs/train/dynamicrafter_512.yaml":
        diffusion_model.model_ema = LitEma(diffusion_model.model)

    # if using non-pretrained model, load the checkpoint
    else:
        if hasattr(eval_config, "act_cond_unet_checkpoint"):
            state_dict = torch.load(eval_config.act_cond_unet_checkpoint)["state_dict"]
            diffusion_model.load_state_dict(state_dict, strict=False)

    diffusion_model.to(device)

    #### Prepare data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    dataset = instantiate_from_config(eval_config.data)
    dataset.setup()
    dataloader = dataset.val_dataloader()

    #### Evaluate >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    sampler = DDIMSampler(diffusion_model)
    evaluate_and_log(eval_config, diffusion_model, device, sampler, dataloader, eval_config.num_batches)
