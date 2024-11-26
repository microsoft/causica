import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger

from fip.my_cli import MyLightningCLI


class ExplicitLogDirTrainer(Trainer):
    """Trainer that uses the log_dir passed to it instead of the one from the config.
    It has been copied from: research_experiments/proxy_tune/entrypoint.py
    """

    def __init__(self, log_dir: str, *args, **kwargs):
        self._log_dir = log_dir
        super().__init__(*args, **kwargs)

    @property
    def log_dir(self) -> str:
        return self._log_dir


def main():
    cli = MyLightningCLI(
        model_class=pl.LightningModule,
        datamodule_class=pl.LightningDataModule,
        trainer_class=ExplicitLogDirTrainer,
        subclass_mode_data=True,
        subclass_mode_model=True,
        save_config_kwargs={"overwrite": True},
        run=False,
    )
    cli.trainer.logger = MLFlowLogger()
    cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path="last")
    cli.trainer.validate(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
