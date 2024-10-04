import os

import tensorflow as tf
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from ldwma.datasets.rtx import get_rtx_tf_dataset
from ldwma.datasets.utils import TensorFlowDatasetWrapper


class RTXDataModule(LightningDataModule):
    """PyTorch Lightning DataModule for the RTX dataset."""

    def __init__(
        self,
        batch_size: int = 32,
        target_height: int = 320,
        target_width: int = 512,
        pad: bool = True,
        deterministic: bool = False,  # dataloading is much slower if true
        use_language: bool = False,
        train_split: str = "train[:95%]",
        val_split: str = "train[95%:]",
        dataset_name: str = "fractal20220817_data",
        dataset_dir: str = "gs://gresearch/robotics",
        save_statistics_dir: str = os.path.join(os.path.dirname(__file__), "octo-statistics"),
        shuffle_buffer: int = 1000,
        traj_len: int = 10,
        num_workers: int = 0,
        pin_memory: bool = True,
        image_key: str = "image_primary",
        seed: int = 0,
        downsample: int = 1,
    ) -> None:
        """Initialize the data module.

        Args:
            batch_size: batch size of all dataloaders
            target_height: The target height of the resized video
            target_width: The target width of the resized video
            pad: whether to pad the images rather than center crop
            train: Whether to load the training or validation set
            dataset_name: The name of the dataset
            dataset_dir: The directory containing the dataset
            save_statistics_dir: The directory that cached statistics are saved to
            shuffle_buffer: The size of the shuffle buffer
            traj_len: The length of each trajectory
            num_workers: number of workers for the torch dataloader
        """
        super().__init__()
        self.batch_size = batch_size
        self.target_height = target_height
        self.target_width = target_width
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.save_statistics_dir = save_statistics_dir
        self.shuffle_buffer = shuffle_buffer
        self.traj_len = traj_len
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pad = pad
        self.train_split = train_split
        self.val_split = val_split
        self.image_key = image_key
        self.downsample = downsample
        self.use_language = use_language
        self.deterministic = deterministic

        rank = self.trainer.global_rank if self.trainer else 0  # Get the rank
        self.seed = seed + rank  # Add the rank to the seed to seed dataloaders for each GPU differently
        tf.random.set_seed(self.seed)

    def setup(self, stage=None):
        pass

    def get_rtx_dataloader(self, train: bool) -> DataLoader:
        """Return the RTX dataset as a PyTorch dataset.

        Args:
            train: Whether to load the training or validation set
        """
        tf_dataset, dataset_len = get_rtx_tf_dataset(
            train=train,
            seed=self.seed,
            deterministic=self.deterministic,
            train_split=self.train_split,
            val_split=self.val_split,
            dataset_name=self.dataset_name,
            dataset_dir=self.dataset_dir,
            save_statistics_dir=self.save_statistics_dir,
            pad=self.pad,
            target_height=self.target_height,
            target_width=self.target_width,
            shuffle_buffer=self.shuffle_buffer,
            traj_len=self.traj_len,
            image_key=self.image_key,
            downsample=self.downsample,
            use_language=self.use_language,
        )
        torch_dataset = TensorFlowDatasetWrapper(tf_dataset, dataset_len)
        dataloader = DataLoader(
            torch_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self.get_rtx_dataloader(train=True)

    def val_dataloader(self) -> DataLoader:
        return self.get_rtx_dataloader(train=False)
