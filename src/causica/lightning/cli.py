import os

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.cli import LightningCLI


class LightningCLIWithDefaults(LightningCLI):
    default_logger = {
        "class_path": "causica.lightning.loggers.BufferingMlFlowLogger",
        "init_args": {"run_id": os.environ.get("AZUREML_RUN_ID", None), "buffer_size": 2000},
    }

    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, "best_checkpoint_callback")
        parser.add_lightning_class_args(ModelCheckpoint, "last_checkpoint_callback")
        parser.link_arguments("best_checkpoint_callback.dirpath", "last_checkpoint_callback.dirpath")

        parser.set_defaults({"trainer.logger": self.default_logger})
