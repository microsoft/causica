from pytorch_lightning.loggers import WandbLogger


def get_run_logger(config, run_kwargs):

    # add the special kwargs to the config
    for key, value in run_kwargs.items():
        setattr(config, key, value)

    # add the special kwargs to the run name
    run_name = config.wandb_name
    if run_kwargs:
        run_name += "_" + "_".join([f"{k}={v}" for k, v in run_kwargs.items()])

    return WandbLogger(
        name=run_name,
        group=config.wandb_group,
        project=config.trainer.logger.init_args.project,
        entity=config.trainer.logger.init_args.entity,
        offline=config.trainer.logger.init_args.offline,
        save_dir=config.trainer.logger.init_args.save_dir,
        config=config,
    )
