from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_lightning_class_args(ModelCheckpoint, "best_checkpoint_callback")
        parser.add_lightning_class_args(ModelCheckpoint, "last_checkpoint_callback")
        parser.add_lightning_class_args(EarlyStopping, "early_stopping_callback")
        parser.link_arguments("best_checkpoint_callback.dirpath", "last_checkpoint_callback.dirpath")
        parser.link_arguments("best_checkpoint_callback.dirpath", "trainer.log_dir")
