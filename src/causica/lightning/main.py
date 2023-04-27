import logging

from causica.lightning.callbacks import MLFlowSaveConfigCallback
from causica.lightning.cli import LightningCLIWithDefaults
from causica.lightning.data_modules.deci_data_module import DECIDataModule
from causica.lightning.modules.deci_module import DECIModule

if __name__ == "__main__":
    # Set Azure logging to warning to prevent spam from HTTP requests
    logging.getLogger("azure").setLevel(logging.WARNING)

    cli = LightningCLIWithDefaults(
        DECIModule,
        DECIDataModule,
        run=False,
        save_config_callback=MLFlowSaveConfigCallback,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(cli.model, datamodule=cli.datamodule)
