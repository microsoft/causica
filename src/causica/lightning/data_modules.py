import abc
import json
from functools import partial
from typing import Any, Dict, OrderedDict

import fsspec
import numpy as np
import pytorch_lightning as pl
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from causica.datasets.csuite_data import (
    DataEnum,
    get_categorical_sizes,
    get_csuite_path,
    tensordict_from_variables_metadata,
)
from causica.datasets.tensordict_utils import convert_one_hot, tensordict_shapes


class DECIDataModule(abc.ABC, pl.LightningDataModule):
    """Data Module for use with the `DECILightningModule`.

    Requires variable shapes to be available before the `setup` step.
    """

    @abc.abstractmethod
    def get_variable_shapes(self) -> OrderedDict[str, torch.Size]:
        """Return the shapes for each variable name."""
        pass

    @abc.abstractmethod
    def get_variable_types(self) -> OrderedDict[str, str]:
        """Return the types for each variable name."""


class VariableSpecDataModule(DECIDataModule):
    """Loads training and test data from fully specified paths for `variables.json` formatted data.

    This format assumes the `variables.json` specified all metadata, and that the corresponding CSV files do not have
    any header rows.

    Note:
        Uses `fsspec` to load the data in the paths. To load from a cloud storage location, make sure the relevant
        `fsspec` plugin is available and provide its corresponding scheme. Provide **storage_options for authentication.
        E.g., to load from an Azure storage account, install `adlfs` and use `az://container@storage_account/path`.

    Args:
        train_path: Path to training CSV data.
        test_path: Path to test CSV data.
        variable_path: Path to `variables.json` file.
        batch_size: Batch size for training and test data.
        dataset_name: Name for dataset.
        storage_options: Storage options forwarded to `fsspec` when loading files.
    """

    def __init__(
        self,
        train_path: str,
        test_path: str,
        variables_path: str,
        batch_size: int = 128,
        dataset_name: str = "anonymous_dataset",
        **storage_options: Dict[str, Any]
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.train_path = train_path
        self.test_path = test_path
        self.variables_path = variables_path
        self.batch_size = batch_size
        self.storage_options = storage_options

    def get_variable_shapes(self) -> OrderedDict[str, torch.Size]:
        if self._variable_shapes is None:
            raise ValueError("Tried to get variable group shapes before data was downloaded.")
        return self._variable_shapes

    def get_variable_types(self) -> OrderedDict[str, str]:
        if self._variable_types is None:
            raise ValueError("Tried to get variable group types before data was downloaded.")
        return self._variable_types

    def prepare_data(self):
        fsspec_open = partial(fsspec.open, mode="r", encoding="utf-8", **self.storage_options)
        with fsspec_open(self.variables_path) as f:
            variables_metadata = json.load(f)

        categorical_sizes = get_categorical_sizes(variables_list=variables_metadata["variables"])
        with fsspec_open(self.train_path) as f:
            arr = np.loadtxt(f, delimiter=",")
            dataset_train: TensorDict = tensordict_from_variables_metadata(arr, variables_metadata["variables"])
            self.dataset_train = convert_one_hot(dataset_train, one_hot_sizes=categorical_sizes)

        with fsspec_open(self.test_path) as f:
            arr = np.loadtxt(f, delimiter=",")
            self.dataset_test: TensorDict = tensordict_from_variables_metadata(arr, variables_metadata["variables"])
            self.dataset_test = convert_one_hot(dataset_train, one_hot_sizes=categorical_sizes)

        assert set(self.dataset_train.keys()) == set(
            self.dataset_test.keys()
        ), "the groups_names for the training and test data must match"
        self._variable_shapes = tensordict_shapes(self.dataset_train)
        self._variable_types = {var["group_name"]: var["type"] for var in variables_metadata["variables"]}

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            collate_fn=lambda x: x,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            collate_fn=lambda x: x,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )


class CSuiteDataModule(VariableSpecDataModule):
    """CSuite data module for loading by the datasets name.

    Args:
        dataset_name: Name of the dataset to load.
        batch_size: Batch size for all datasets.
        dataset_path: Path to CSuite dataset mirror.
    """

    DEFAULT_CSUITE_PATH = "https://azuastoragepublic.blob.core.windows.net/datasets"

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 128,
        dataset_path: str = DEFAULT_CSUITE_PATH,
    ):
        super().__init__(
            train_path=get_csuite_path(dataset_path, dataset_name, DataEnum.TRAIN),
            test_path=get_csuite_path(dataset_path, dataset_name, DataEnum.TEST),
            variables_path=get_csuite_path(dataset_path, dataset_name, DataEnum.VARIABLES_JSON),
            batch_size=batch_size,
            dataset_name=dataset_name,
        )
