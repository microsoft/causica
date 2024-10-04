import argparse
import datetime
import logging
import os
import signal

import pudb
from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer

mainlogger = logging.getLogger("mainlogger")

from collections import OrderedDict

import torch
from lvdm.models.ddpm3d import DDPM
from lvdm.utils.utils import instantiate_from_config


def init_workspace(name, logdir, model_config, lightning_config, rank=0):
    workdir = os.path.join(logdir, name)
    ckptdir = os.path.join(workdir, "checkpoints")
    cfgdir = os.path.join(workdir, "configs")
    loginfo = os.path.join(workdir, "loginfo")

    # Create logdirs and save configs (all ranks will do to avoid missing directory error if rank:0 is slower)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(loginfo, exist_ok=True)

    if rank == 0:
        if "callbacks" in lightning_config and "metrics_over_trainsteps_checkpoint" in lightning_config.callbacks:
            os.makedirs(os.path.join(ckptdir, "trainstep_checkpoints"), exist_ok=True)
        OmegaConf.save(model_config, os.path.join(cfgdir, "model.yaml"))
        OmegaConf.save(OmegaConf.create({"lightning": lightning_config}), os.path.join(cfgdir, "lightning.yaml"))
    return workdir, ckptdir, cfgdir, loginfo


def get_env_vars():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    local_rank = int(os.environ.get("LOCAL_RANK"))
    global_rank = int(os.environ.get("RANK"))
    num_rank = int(os.environ.get("WORLD_SIZE"))
    return now, local_rank, global_rank, num_rank


def prepare_logger(lightning_config, config, global_rank, now):
    lightning_config.trainer.logger.params.name = config.name
    lightning_config.trainer.logger.params.group = config.group
    workdir, ckptdir, cfgdir, loginfo = init_workspace(
        config.name, config.logdir, config, lightning_config, global_rank
    )
    logger = set_logger(logfile=os.path.join(loginfo, f"log_{global_rank}:{now}.txt"))
    return logger, workdir, ckptdir, cfgdir, loginfo


def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None


def set_allow_checkpointing(trainer, ckptdir):
    ## allow checkpointing via USR1
    def melk(*args, **kwargs):
        ## run all checkpoint hooks
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last_summoning.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            pudb.set_trace()

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)


def get_trainer(lightning_config, trainer_config, config, args, workdir, ckptdir, logger):
    if "accelerator" not in trainer_config:
        trainer_config["accelerator"] = "gpu"

    ## update trainer config
    for k in get_nondefault_trainer_args(args):
        trainer_config[k] = getattr(args, k)

    ## setup trainer args: pl-logger and callbacks
    trainer_kwargs = dict()
    trainer_kwargs["num_sanity_val_steps"] = 0
    trainer_kwargs["logger"] = instantiate_from_config(trainer_config["logger"])

    ## setup callbacks
    callbacks_cfg = get_trainer_callbacks(lightning_config, config, workdir, ckptdir, logger)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    strategy_cfg = get_trainer_strategy(lightning_config)
    trainer_kwargs["strategy"] = strategy_cfg if type(strategy_cfg) == str else instantiate_from_config(strategy_cfg)
    trainer_kwargs["precision"] = lightning_config.get("precision", 32)
    trainer_kwargs["sync_batchnorm"] = False

    ## trainer config: others
    trainer_args = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(trainer_args, **trainer_kwargs)

    set_allow_checkpointing(trainer, ckptdir)
    return trainer


def get_model(model_config, workdir=None):
    if workdir:
        model_config.params.logdir = workdir
    model = instantiate_from_config(model_config)
    model = load_checkpoints(model, model_config)

    ## register_schedule again to make ZTSNR work
    if isinstance(model, DDPM):
        if model.rescale_betas_zero_snr:
            model.register_schedule(
                given_betas=model.given_betas,
                beta_schedule=model.beta_schedule,
                timesteps=model.timesteps,
                linear_start=model.linear_start,
                linear_end=model.linear_end,
                cosine_s=model.cosine_s,
            )
    return model


def set_model_lr(model, model_config, num_rank, batch_size):
    base_lr = model_config.base_learning_rate
    if getattr(model_config, "scale_lr", False):
        model.learning_rate = num_rank * batch_size * base_lr
        print("Scaling learning rate to: ", model.learning_rate)
    else:
        model.learning_rate = base_lr
    return model


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--seed", "-s", type=int, default=20230211, help="seed for seed_everything")

    parser.add_argument(
        "--base",
        "-b",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )

    parser.add_argument("--train", "-t", action="store_true", default=False, help="train")
    parser.add_argument("--val", "-v", action="store_true", default=False, help="val")
    parser.add_argument("--test", action="store_true", default=False, help="test")
    parser.add_argument("--auto_resume", action="store_true", default=False, help="resume from full-info checkpoint")
    parser.add_argument(
        "--auto_resume_weight_only", action="store_true", default=False, help="resume from weight-only checkpoint"
    )
    parser.add_argument("--debug", "-d", action="store_true", default=False, help="enable post-mortem debugging")

    return parser


def get_nondefault_trainer_args(args):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    default_trainer_args = parser.parse_args([])
    return sorted(k for k in vars(default_trainer_args) if getattr(args, k) != getattr(default_trainer_args, k))


def get_trainer_callbacks(lightning_config, config, logdir, ckptdir, logger):
    default_callbacks_cfg = {
        "model_checkpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}",
                "verbose": True,
                "save_last": False,
            },
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {"logging_interval": "step", "log_momentum": False},
        },
        "cuda_callback": {"target": "lvdm.utils.callbacks.CUDACallback"},
    }

    ## optional setting for saving checkpoints
    monitor_metric = check_config_attribute(config.model.params, "monitor")
    if monitor_metric is not None:
        mainlogger.info(f"Monitoring {monitor_metric} as checkpoint metric.")
        default_callbacks_cfg["model_checkpoint"]["params"]["monitor"] = monitor_metric
        default_callbacks_cfg["model_checkpoint"]["params"]["save_top_k"] = 3
        default_callbacks_cfg["model_checkpoint"]["params"]["mode"] = "min"

    if "metrics_over_trainsteps_checkpoint" in lightning_config.callbacks:
        mainlogger.info(
            "Caution: Saving checkpoints every n train steps without deleting. This might require some free space."
        )
        default_metrics_over_trainsteps_ckpt_dict = {
            "metrics_over_trainsteps_checkpoint": {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                    "filename": "{epoch}-{step}",
                    "verbose": True,
                    "save_top_k": -1,
                    "every_n_train_steps": 10000,
                    "save_weights_only": True,
                },
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    return callbacks_cfg


def get_trainer_logger(lightning_config, logdir, on_debug):
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": "tensorboard",
            },
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            },
        },
    }
    os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
    default_logger_cfg = default_logger_cfgs["tensorboard"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg


def get_trainer_strategy(lightning_config):
    default_strategy_dict = {"target": "pytorch_lightning.strategies.DDPShardedStrategy"}
    if "strategy" in lightning_config:
        strategy_cfg = lightning_config.strategy
        return strategy_cfg
    else:
        strategy_cfg = OmegaConf.create()

    strategy_cfg = OmegaConf.merge(default_strategy_dict, strategy_cfg)
    return strategy_cfg


def load_checkpoints(model, model_cfg):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), "Error: Pre-trained checkpoint NOT found at:%s" % pretrained_ckpt
        mainlogger.info(">>> Load weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        if "state_dict" in pl_sd.keys():
            state_dict = pl_sd["state_dict"]
        else:
            # deepspeed
            state_dict = OrderedDict()
            for key in pl_sd["module"].keys():
                state_dict[key[16:]] = pl_sd["module"][key]

        if check_config_attribute(model_cfg, "only_reload_modules"):
            for prefix in model_cfg.only_reload_modules:
                if not any(k.startswith(prefix) for k in state_dict.keys()):
                    raise ValueError(f"No modules found starting with: {prefix}")

            include_prefixes = tuple(model_cfg.only_reload_modules)
            state_dict = {k: v for k, v in state_dict.items() if k.startswith(include_prefixes)}
            mainlogger.info(">>> Reloading only the modules: %s" % model_cfg.only_reload_modules)

        model.load_state_dict(state_dict, strict=False)
        mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s" % pretrained_ckpt)
    else:
        mainlogger.info(">>> Start training from scratch")

    return model


def set_logger(logfile, name="mainlogger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode="w")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
