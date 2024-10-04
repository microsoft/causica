import json

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI, Namespace, SaveConfigCallback
from pytorch_lightning.loggers import WandbLogger


class SaveConfigToWandbCallback(SaveConfigCallback):
    """Adds config to the Weights and Biases run via callback.

    Standard WandbLogger does not save the configuration file, so we need to do it manually.
    """

    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Namespace,
        config_filename: str = "config.yaml",
        multifile: bool = False,
    ) -> None:
        super().__init__(
            parser=parser, config=config, config_filename=config_filename, overwrite=True, multifile=multifile
        )

    def save_config(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        # Save the file on rank 0
        if isinstance(trainer.logger, WandbLogger):
            config = json.loads(
                self.parser.dump(self.config, skip_none=False, format="json")
            )  # Required for proper reproducibility
            trainer.logger.experiment.config.update(config)


class LightningCLIDiffusion(LightningCLI):
    def __init__(self, *args, **kwargs) -> None:
        if "save_config_callback" not in kwargs:
            kwargs["save_config_callback"] = SaveConfigToWandbCallback
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("--wandb_group", default="devel", type=str)
        parser.add_argument("--wandb_name", default="diffusion", type=str)
        parser.add_argument("--checkpoint_path", default="None", type=str)
        parser.add_argument("--num_batches", default=0, type=int)

        parser.link_arguments(
            "model.init_args.condition_frames",
            "model.init_args.diffusion_model.init_args.model.init_args.condition_frames",
        )
        parser.link_arguments(
            "model.init_args.num_pred_frames",
            "model.init_args.diffusion_model.init_args.model.init_args.generation_frames",
        )
        parser.link_arguments(
            "wandb_group",
            "trainer.logger.init_args.group",
        )
        parser.link_arguments(
            "wandb_name",
            "trainer.logger.init_args.name",
        )


class MixedDiffusionCLI(LightningCLI):
    def __init__(self, *args, **kwargs) -> None:
        if "save_config_callback" not in kwargs:
            kwargs["save_config_callback"] = SaveConfigToWandbCallback
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("--wandb_group", default="devel", type=str)
        parser.add_argument("--wandb_name", default="diffusion", type=str)
        parser.add_argument("--act_cond_wts", type=list)
        parser.add_argument("--no_act_cond_steps", type=list)
        parser.add_argument("--base_model_ckpt", type=str)
        parser.add_argument("--act_cond_model_ckpt", type=str)
        parser.add_argument("--num_samples", type=int)
        parser.add_argument("--sampling_frames", type=int)
        parser.add_argument("--num_batches", type=int)
        parser.add_argument("--cfg_type", type=str)

        parser.link_arguments(
            "wandb_group",
            "trainer.logger.init_args.group",
        )
        parser.link_arguments(
            "wandb_name",
            "trainer.logger.init_args.name",
        )


class ClassifierCLI(LightningCLI):
    def __init__(self, *args, **kwargs) -> None:
        if "save_config_callback" not in kwargs:
            kwargs["save_config_callback"] = SaveConfigToWandbCallback
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument("--wandb_group", default="devel", type=str)
        parser.add_argument("--wandb_name", default="diffusion", type=str)
        parser.link_arguments(
            "wandb_group",
            "trainer.logger.init_args.group",
        )
        parser.link_arguments(
            "wandb_name",
            "trainer.logger.init_args.name",
        )


class LightningCLIAvid(LightningCLIDiffusion):
    def add_arguments_to_parser(self, parser) -> None:
        super().add_arguments_to_parser(parser)

        parser.link_arguments(
            "model.init_args.condition_frames",
            "model.init_args.diffusion_model.init_args.adapter.init_args.condition_frames",
        )
        parser.link_arguments(
            "model.init_args.num_pred_frames",
            "model.init_args.diffusion_model.init_args.adapter.init_args.generation_frames",
        )

    def before_instantiate_classes(self):
        args = self.config

        # Add additional conditioning frame to adapter if it is conditioning on the base model output
        if args.model.init_args.diffusion_model.init_args.condition_adapter_on_base_outputs:
            args.model.init_args.diffusion_model.init_args.adapter.init_args.condition_frames += 1

        if args.model.init_args.diffusion_model.init_args.learnt_mask:
            args.model.init_args.diffusion_model.init_args.adapter.init_args.output_mask = True

        # Optionally call the original method if needed
        super().before_instantiate_classes()


class LightningCLIControlNet(LightningCLIDiffusion):
    def add_arguments_to_parser(self, parser) -> None:
        super().add_arguments_to_parser(parser)
        parser.link_arguments(
            "model.init_args.condition_frames",
            "model.init_args.diffusion_model.init_args.control_net.init_args.condition_frames",
        )
        parser.link_arguments(
            "model.init_args.num_pred_frames",
            "model.init_args.diffusion_model.init_args.control_net.init_args.generation_frames",
        )
