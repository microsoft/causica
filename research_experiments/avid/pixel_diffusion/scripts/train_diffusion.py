import torch

from dwma.lightning.cli.cli import LightningCLIDiffusion
from dwma.lightning.data_modules.procgen_data import ProcgenDataModule
from dwma.lightning.modules.diffusion_module import DiffusionModule

if __name__ == "__main__":
    cli = LightningCLIDiffusion(
        model_class=DiffusionModule,
        datamodule_class=ProcgenDataModule,
        run=False,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )

    # for finetuning base model
    if cli.config.checkpoint_path != "None":
        checkpoint = torch.load(cli.config.checkpoint_path, map_location="cpu")
        cli.model.load_state_dict(checkpoint["state_dict"])

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
