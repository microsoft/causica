from dwma.lightning.cli.cli import LightningCLIAvid
from dwma.lightning.data_modules.procgen_data import ProcgenDataModule
from dwma.lightning.modules.diffusion_module import DiffusionModule

if __name__ == "__main__":
    cli = LightningCLIAvid(
        model_class=DiffusionModule,
        datamodule_class=ProcgenDataModule,
        run=False,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
