import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.cli import LightningCLI


def main():
    cli = LightningCLI(
        model_class=pl.LightningModule,
        datamodule_class=pl.LightningDataModule,
        trainer_class=Trainer,
        subclass_mode_data=True,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
        run=False,
    )
    cli.trainer.test(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
