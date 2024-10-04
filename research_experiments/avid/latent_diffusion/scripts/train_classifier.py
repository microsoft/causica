from lvdm.utils.train import get_env_vars, get_model, get_parser, get_trainer, prepare_logger, set_model_lr
from lvdm.utils.utils import instantiate_from_config
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

if __name__ == "__main__":
    now, local_rank, global_rank, num_rank = get_env_vars()

    # Extends existing argparse by default Trainer attributes
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    seed_everything(args.seed)

    # yaml configs
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.get("trainer", OmegaConf.create())

    # setup workspace directories and logger
    logger, workdir, ckptdir, cfgdir, loginfo = prepare_logger(lightning_config, config, global_rank, now)

    ## CLASSIFIER MODEL CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    classifier = get_model(config.model, workdir)
    classifier = set_model_lr(classifier, config.model, num_rank, config.data.params.batch_size)

    ## DIFFUSION MODEL FOR ENCODING AND NOISING DATA CONFIG >>>>>>>>>>>>>>>>>>>>>>>>
    if hasattr(config, "diffusion_config_file"):
        diffusion_config = OmegaConf.load(config.diffusion_config_file).model
        diffusion_model = instantiate_from_config(diffusion_config)
        classifier.set_diffusion_model(diffusion_model)

    ## DATA CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    data = instantiate_from_config(config.data)
    data.setup()

    ## TRAINER CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    trainer = get_trainer(
        lightning_config=lightning_config,
        trainer_config=trainer_config,
        config=config,
        args=args,
        workdir=workdir,
        ckptdir=ckptdir,
        logger=logger,
    )

    ## TRAINING >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    trainer.fit(classifier, data)
