"""Lightning Classes for loading data in the default format used in Azure Blob Storage."""
import os
from collections import defaultdict
from functools import partial
from typing import Any, Optional

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from causica.datasets.causica_dataset_format import DataEnum, load_data
from causica.datasets.standardizer import JointStandardizer, fit_standardizer
from causica.datasets.tensordict_utils import identity, tensordict_shapes
from causica.datasets.variable_types import VariableTypeEnum
from causica.lightning.data_modules.deci_data_module import DECIDataModule


class VariableSpecDataModule(DECIDataModule):
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
        **storage_options: dict[str, Any],
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
        self._dataset_name = dataset_name
        self.root_path = root_path
        self.batch_size = batch_size
        self.storage_options = storage_options
        self.normalize = normalize
        self.load_counterfactual = load_counterfactual

        self.normalizer: Optional[JointStandardizer] = None

    @property
    def variable_shapes(self) -> dict[str, torch.Size]:
        return _check_exists(self, "_variable_shapes")

    @property
    def variable_types(self) -> dict[str, VariableTypeEnum]:
        return _check_exists(self, "_variable_types")

    @property
    def column_names(self) -> dict[str, list[str]]:
        return _check_exists(self, "_column_names")

    @property
    def dataset_train(self) -> TensorDict:
        return _check_exists(self, "_dataset_train")

    @property
    def dataset_test(self) -> TensorDict:
        return _check_exists(self, "_dataset_test")

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def _load_all_data(self, variables_metadata: dict):
        _load_data = partial(
            load_data, root_path=self.root_path, variables_metadata=variables_metadata, **self.storage_options
        )

        self._dataset_train = _load_data(data_enum=DataEnum.TRAIN)
        self._dataset_test = _load_data(data_enum=DataEnum.TEST)
        self.true_adj = _load_data(data_enum=DataEnum.TRUE_ADJACENCY)
        self.interventions = _load_data(data_enum=DataEnum.INTERVENTIONS)

        self.counterfactuals = None
        if self.load_counterfactual:
            self.counterfactuals = _load_data(data_enum=DataEnum.COUNTERFACTUALS)

    def prepare_data(self):
        # WARNING: Do not remove partial here. For some reason, if `load_data` is called directly from inside this data
        # module without being wrapped in partial, it's first optional `variables_metadata` argument is added to the
        # config values of this class. Upon init through the CLI, this argument then becomes interpreted as a value in
        # `storage_options`, whereby other calls to `load_data` will fail due to having two keyword arguments named
        # `variables_metadata`.
        #
        # I.e. if `load_data` is called directly from here, the lightning command line with this class as a data module
        # and `--print_config` produces:
        # ...
        # data:
        #  class_path: causica.lightning.data_modules.VariableSpecDataModule
        #  init_args:
        #    ...
        #    variables_metadata: null
        _load_data = partial(load_data, root_path=self.root_path, **self.storage_options)
        variables_metadata = _load_data(data_enum=DataEnum.VARIABLES_JSON)

        self._load_all_data(variables_metadata)

        train_keys = set(self._dataset_train.keys())
        test_keys = set(self._dataset_test.keys())
        assert (
            train_keys == test_keys
        ), f"node_names for the training and test data must match. Diff: {train_keys.symmetric_difference(test_keys)}"
        self._variable_shapes = tensordict_shapes(self._dataset_train)
        self._variable_types = {var["group_name"]: var["type"] for var in variables_metadata["variables"]}
        self._column_names = defaultdict(list)
        for variable in variables_metadata["variables"]:
            self._column_names[variable["group_name"]].append(variable["name"])

        if self.normalize:
            continuous_keys = [k for k, v in self._variable_types.items() if v == VariableTypeEnum.CONTINUOUS]
            self.normalizer = fit_standardizer(self._dataset_train.select(*continuous_keys))

            transform = self.normalizer()
            self._dataset_train = transform(self._dataset_train)
            self._dataset_test = transform(self._dataset_test)

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


def _check_exists(obj: Any, attribute_name: str):
    """Check if an attribute exists otherwise print a message."""
    try:
        return getattr(obj, attribute_name)
    except AttributeError as exc:
        display_string = attribute_name.replace("_", " ").strip()
        raise ValueError(f"Tried to get {display_string} before data was downloaded.") from exc
