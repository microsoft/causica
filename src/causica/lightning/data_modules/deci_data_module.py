import abc

import pytorch_lightning as pl
import torch
from tensordict import TensorDict

from causica.datasets.variable_types import VariableTypeEnum


class DECIDataModule(pl.LightningDataModule, abc.ABC):
    """An Abstract Data Module containing the methods required by a `DECIModule`."""

    @property
    @abc.abstractmethod
    def dataset_name(self) -> str:
        """The name of this dataset"""

    @property
    @abc.abstractmethod
    def dataset_train(self) -> TensorDict:
        """The training dataset"""

    @property
    @abc.abstractmethod
    def dataset_test(self) -> TensorDict:
        """The testing dataset"""

    @property
    @abc.abstractmethod
    def variable_shapes(self) -> dict[str, torch.Size]:
        """Get the shape of each variable in the dataset."""

    @property
    @abc.abstractmethod
    def variable_types(self) -> dict[str, VariableTypeEnum]:
        """Get the type of each variable in the dataset."""

    @property
    @abc.abstractmethod
    def column_names(self) -> dict[str, list[str]]:
        """Get a map of the node names and the corresponding columns of the original dataset."""
