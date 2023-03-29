"""Lightning Classes for loading data in the default format used in Azure Blob Storage."""
import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from causica.datasets.causica_dataset_format import DataEnum, load_data
from causica.datasets.standardizer import JointStandardizer, fit_standardizer
from causica.datasets.tensordict_utils import identity, tensordict_shapes
from causica.datasets.variable_types import VariableTypeEnum


class VariableSpecDataModule(pl.LightningDataModule):
    """
    Loads training and test data from fully specified paths for `variables.json` formatted data.

    This format assumes the `variables.json` specified all metadata, and that the corresponding CSV files do not have
    any header rows.

    Note:
        Uses `fsspec` to load the data in the paths. To load from a cloud storage location, make sure the relevant
        `fsspec` plugin is available and provide its corresponding scheme. Provide **storage_options for authentication.
        E.g., to load from an Azure storage account, install `adlfs` and use `az://container@storage_account/path`.

    """

    def __init__(
        self,
        root_path: str,
        batch_size: int = 128,
        dataset_name: str = "anonymous_dataset",
        normalize: bool = False,
        load_counterfactual: bool = False,
        **storage_options: Dict[str, Any],
    ):
        """
        Args:
            root_path: Path to directory with causal data
            batch_size: Batch size for training and test data.
            storage_options: Storage options forwarded to `fsspec` when loading files.
            dataset_name: A name for the dataset
            load_counterfactual: Whether there is counterfactual data
            normalize: Whether to normalize the data
        """
        super().__init__()
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.root_path = root_path
        self.batch_size = batch_size
        self.storage_options = storage_options
        self.normalize = normalize
        self.load_counterfactual = load_counterfactual

        self.normalizer: Optional[JointStandardizer] = None

        # add type hints
        self.dataset_train: TensorDict
        self.dataset_test: TensorDict

    def get_variable_shapes(self) -> Dict[str, torch.Size]:
        if self._variable_shapes is None:
            raise ValueError("Tried to get variable group shapes before data was downloaded.")
        return self._variable_shapes

    def get_variable_types(self) -> Dict[str, VariableTypeEnum]:
        if self._variable_types is None:
            raise ValueError("Tried to get variable group types before data was downloaded.")
        return self._variable_types

    def get_variable_names(self) -> Dict[str, str]:
        if self._variable_names is None:
            raise ValueError("Tried to get variable group shapes before data was downloaded.")
        return self._variable_names

    def _load_all_data(self, variables_metadata: Dict):
        _load_data = partial(
            load_data, root_path=self.root_path, variables_metadata=variables_metadata, **self.storage_options
        )

        self.dataset_train = _load_data(data_enum=DataEnum.TRAIN)
        self.dataset_test = _load_data(data_enum=DataEnum.TEST)
        self.true_adj = _load_data(data_enum=DataEnum.TRUE_ADJACENCY)
        self.interventions = _load_data(data_enum=DataEnum.INTERVENTIONS)

        self.counterfactuals = None
        if self.load_counterfactual:
            self.counterfactuals = _load_data(data_enum=DataEnum.COUNTERFACTUALS)

    def prepare_data(self):
        variables_metadata = load_data(self.root_path, DataEnum.VARIABLES_JSON, **self.storage_options)

        self._load_all_data(variables_metadata)

        train_keys = set(self.dataset_train.keys())
        test_keys = set(self.dataset_test.keys())
        assert (
            train_keys == test_keys
        ), f"The node_names for the training and test data must match. Diff: {train_keys.symmetric_difference(test_keys)}"
        self._variable_shapes = tensordict_shapes(self.dataset_train)
        self._variable_types = {var["group_name"]: var["type"] for var in variables_metadata["variables"]}
        self._variable_names = defaultdict(list)
        for v in variables_metadata["variables"]:
            self._variable_names["group_name"].append(v["name"])

        if self.normalize:
            continuous_keys = [k for k, v in self._variable_types.items() if v == VariableTypeEnum.CONTINUOUS]
            self.normalizer = fit_standardizer(self.dataset_train.select(*continuous_keys))

            transform = self.normalizer()
            self.dataset_train = transform(self.dataset_train)
            self.dataset_test = transform(self.dataset_test)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            collate_fn=identity,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )

    def test_dataloader(self):
        test_dataloader = DataLoader(dataset=self.dataset_test, collate_fn=identity, batch_size=self.batch_size)

        dataloader_list = [
            test_dataloader,
            DataLoader(dataset=self.true_adj[None, ...]),
            DataLoader(dataset=self.interventions, collate_fn=identity, batch_size=None),
        ]

        if self.counterfactuals is not None:
            dataloader_list.append(DataLoader(dataset=self.counterfactuals, collate_fn=identity, batch_size=None))

        return dataloader_list


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
        load_counterfactual: bool = False,
    ):
        super().__init__(
            root_path=os.path.join(dataset_path, dataset_name),
            batch_size=batch_size,
            dataset_name=dataset_name,
            load_counterfactual=load_counterfactual,
        )
