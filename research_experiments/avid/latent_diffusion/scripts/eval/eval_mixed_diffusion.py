import argparse

import torch
from lvdm.ema import LitEma
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.utils.eval import evaluate_and_log
from lvdm.utils.train import get_model
from lvdm.utils.utils import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from ldwma.models.wrappers.mixed_diffusion import MixedDiffusionWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=0, help="seed for seed_everything")
    parser.add_argument("--config", "-e", type=str, help="path to config file")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda:0")

    eval_config = OmegaConf.load(args.config)
    base_model_config = OmegaConf.load(eval_config.base_config_file).model
    action_model_config = OmegaConf.load(eval_config.action_config_file).model

    # use the checkpoints from the eval config
    base_model_config.pretrained_checkpoint = eval_config.base_model_checkpoint
    action_model_config.pretrained_checkpoint = eval_config.action_model_checkpoint

    # ensure that the entirety of both models is reloaded from checkpoint
    base_model_config.only_reload_modules = None
    action_model_config.only_reload_modules = None

    # instantiate both of the models using the configs
    base_model = get_model(base_model_config)
    base_model.model_ema = LitEma(base_model.model)
    action_model = get_model(action_model_config)
    base_model.to(device)
    action_model.to(device)

    # iterate over different inferece params
    for no_act_cond_steps in eval_config.no_act_cond_steps_list:
        for act_cond_wt in eval_config.act_cond_wt_list:

            # load the validation dataset using the args in the eval_config
            dataset = instantiate_from_config(eval_config.data)
            dataset.setup()
            dataloader = dataset.val_dataloader()

            # make mixed diffusion model with inference params
            mixed_diffusion = MixedDiffusionWrapper(
                base_model=base_model,
                action_conditioned_model=action_model,
                mixing_strategy=eval_config.mixing_strategy,
                act_cond_wt=act_cond_wt,
                no_act_cond_steps=no_act_cond_steps,
                linear_decay=eval_config.linear_decay,
            )
            sampler = DDIMSampler(mixed_diffusion)
            with action_model.ema_scope("Plotting"):  # ensures ema of both models used
                evaluate_and_log(
                    eval_config,
                    base_model,
                    device,
                    sampler,
                    dataloader,
                    eval_config.num_batches,
                    run_kwargs={"act_cond_wt": act_cond_wt, "no_act_cond_stp": no_act_cond_steps},
                )
