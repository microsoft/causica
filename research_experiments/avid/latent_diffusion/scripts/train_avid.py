from lvdm.models.ddpm3d import DiffusionWrapper
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

    # load the configs for the two models
    base_model_config = OmegaConf.load(config.base_config_file).model
    action_model_config = OmegaConf.load(config.action_config_file).model

    # setup workspace directories and logger
    logger, workdir, ckptdir, cfgdir, loginfo = prepare_logger(lightning_config, config, global_rank, now)

    ## MODEL CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # configure number of input channels for the action model
    if config.adapter_params.get("condition_adapter_on_base_outputs", False):
        action_model_config.params.unet_config.params.in_channels += (
            base_model_config.params.unet_config.params.out_channels
        )

    # configure whether to output mask from the action model
    if config.adapter_params.get("learnt_mask", False):
        action_model_config.params.unet_config.params.output_mask = True

    # change target module for base_model to avid module
    base_model_config.target = config.target_module
    avid_model = get_model(base_model_config, workdir)
    avid_model = set_model_lr(avid_model, base_model_config, num_rank, config.data.params.batch_size)

    # create action-conditioned unet
    action_cond_unet = DiffusionWrapper(
        action_model_config.params.unet_config, conditioning_key=action_model_config.params.conditioning_key
    )

    # add the action-conditioned unet to the avid model
    avid_model.prepare_adapter(action_cond_unet, **config.adapter_params)

    ## DATA CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    data = instantiate_from_config(config.data)
    data.setup()

    ## TRAINER CONFIG >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    config.model = base_model_config  # for setting up the trainer args
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
    trainer.fit(avid_model, data)
