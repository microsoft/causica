import pytorch_lightning as pl
from taming.data.custom import CustomTest, CustomTrain
from torch.utils.data import DataLoader


class DataModuleFromConfig(pl.LightningDataModule):
    """Loads data from files.

    Sets up a dataloader based on data for loading taming transformers datasets such as coco.
    """

    def __init__(
        self,
        batch_size,
        train: CustomTrain | None = None,
        validation: CustomTest | None = None,
        test: CustomTest | None = None,
        num_workers=None,
    ):
        """
        Args:
            batch_size: Batch size for the dataloader.
            train: Training dataset.
            validation: Validation dataset.
            test: Test dataset.
            num_workers: Number of workers for the dataloader.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.datasets = {}
        if train is not None:
            self.datasets["train"] = train
        if validation is not None:
            self.datasets["validation"] = validation
        if test is not None:
            self.datasets["test"] = test

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.datasets["validation"], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size, num_workers=self.num_workers)
