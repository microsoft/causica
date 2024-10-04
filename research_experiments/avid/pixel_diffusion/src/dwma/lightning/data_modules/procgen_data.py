import os

from pytorch_lightning import LightningDataModule

from dwma.datasets.procgen_dataset import ProcgenDataLoader, ProcgenDataset


class ProcgenDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for Procgen data."""

    def __init__(
        self,
        batch_size: int = 32,
        train_data_folder: str = "az://fvwm@azuastorage.blob.core.windows.net/genie/data/coinrun_train",
        test_data_folder: str = "az://fvwm@azuastorage.blob.core.windows.net/genie/data/coinrun_test",
        val_data_folder: str = "az://fvwm@azuastorage.blob.core.windows.net/genie/data/coinrun_val",
        num_workers: int = 24,
        cache_files: bool = True,
        precache_files: bool = False,
        cache_dir: str = str(os.path.expanduser("~/.cache/procgen")),
        fixed_episode_length: int | None = 1000,
        window_width: int = 15,
    ) -> None:
        """Initialize the data module.

        Args:
            batch_size: batch size of all dataloaders
            train_data_folder: folder containing the procgen training data
            test_data_folder: folder containing the procgen test data
            val_data_folder: folder containing the procgen validation data
            num_workers: number of workers to use for all dataloaders
            cache_files: whether to cache the files locally
            cache_dir: directory to store the cached files
            fixed_episode_length (int | None): known fixed length of each episode (if None, the actual length of each
                episode is first retrieved, but this may be slow)
            window_width (int): width of the window used to generate examples. For example, the observations in each
                batch will be of shape (batch_size, window_width, height, width, channels).
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_data_folder = train_data_folder
        self.test_data_folder = test_data_folder
        self.val_data_folder = val_data_folder
        self.num_workers = num_workers
        self.cache_files = cache_files
        self.cache_dir = cache_dir
        self.fixed_episode_length = fixed_episode_length
        self.window_width = window_width
        self.precache_files = precache_files

    def get_procgen_dataset(self, data_folder: str) -> ProcgenDataset:
        """Return a ProcgenDataset object for the given data folder.

        Args:
            data_folder: folder containing the Procgen data

        Returns:
            ProcgenDataset object
        """
        return ProcgenDataset(
            data_folder=data_folder,
            window_width=self.window_width,
            fixed_episode_length=self.fixed_episode_length,
            cache_files=self.cache_files,
            cache_dir=self.cache_dir,
            precache_files=self.precache_files,
        )

    def train_dataloader(self) -> ProcgenDataLoader:
        dataset = self.get_procgen_dataset(self.train_data_folder)
        return ProcgenDataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def test_dataloader(self) -> ProcgenDataLoader:
        dataset = self.get_procgen_dataset(self.test_data_folder)
        return ProcgenDataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self) -> ProcgenDataLoader:
        dataset = self.get_procgen_dataset(self.val_data_folder)
        return ProcgenDataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
