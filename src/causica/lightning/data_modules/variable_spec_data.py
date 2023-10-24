"""Lightning Classes for loading data in the default format used in Azure Blob Storage."""
import os
from collections import defaultdict
from functools import partial
from typing import Any, Iterable, Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase
from torch.utils.data import DataLoader

from causica.datasets.causica_dataset_format import CAUSICA_DATASETS_PATH, DataEnum, VariablesMetadata, load_data
from causica.datasets.normalization import FitNormalizerType, Normalizer, fit_standardizer
from causica.datasets.tensordict_utils import identity, tensordict_shapes
from causica.datasets.variable_types import VariableTypeEnum
from causica.distributions.transforms import JointTransformModule
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
        normalize: Union[bool, Iterable[str]] = False,
        exclude_normalization: Iterable[str] = tuple(),
        fit_normalizer: FitNormalizerType = fit_standardizer,
        load_counterfactual: bool = False,
        load_interventional: bool = False,
        **storage_options: Any,
    ):
        """
        Args:
            root_path: Path to directory with causal data
            batch_size: Batch size for training and test data.
            dataset_name: A name for the dataset
            normalize: Whether to normalize the data or list of variables to normalize
            exclude_normalization: Which variables to exclude from normalization
            load_counterfactual: Whether counterfactual data should be loaded
            load_interventional: Whether interventional data should be loaded
            **storage_options: Storage options forwarded to `fsspec` when loading files.
        """
        super().__init__()
        self.batch_size = batch_size
        self._dataset_name = dataset_name
        self.root_path = root_path
        self.batch_size = batch_size
        self.storage_options = storage_options
        self.normalize = normalize
        self.exclude_normalization = set(exclude_normalization)
        self.load_counterfactual = load_counterfactual
        self.load_interventional = load_interventional

        self.fit_normalizer = fit_normalizer
        self.normalizer: Optional[Normalizer] = None
        self._dataset_train: TensorDictBase
        self._dataset_test: TensorDictBase
        self.true_adj: torch.Tensor

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

    def _load_all_data(self, variables_metadata: VariablesMetadata):
        _load_data = partial(
            load_data, root_path=self.root_path, variables_metadata=variables_metadata, **self.storage_options
        )

        dataset_train = _load_data(data_enum=DataEnum.TRAIN)
        dataset_test = _load_data(data_enum=DataEnum.TEST)
        true_adj = _load_data(data_enum=DataEnum.TRUE_ADJACENCY)
        assert isinstance(dataset_train, TensorDict)
        assert isinstance(dataset_test, TensorDict)
        assert isinstance(true_adj, torch.Tensor)
        self._dataset_train = dataset_train
        self._dataset_test = dataset_test
        self.true_adj = true_adj

        self.interventions = []
        if self.load_interventional:
            self.interventions = _load_data(data_enum=DataEnum.INTERVENTIONS)

        self.counterfactuals = []
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
        _load_data = partial(load_data, root_path=self.root_path, **self.storage_options)  # type: ignore
        variables_metadata: VariablesMetadata = _load_data(data_enum=DataEnum.VARIABLES_JSON)  # type: ignore

        self._load_all_data(variables_metadata)

        train_keys = set(self._dataset_train.keys())
        test_keys = set(self._dataset_test.keys())
        assert (
            train_keys == test_keys
        ), f"node_names for the training and test data must match. Diff: {train_keys.symmetric_difference(test_keys)}"
        self._variable_shapes = tensordict_shapes(self._dataset_train)
        self._variable_types = {var.group_name: var.type for var in variables_metadata.variables}
        self._column_names = defaultdict(list)
        for variable in variables_metadata.variables:
            self._column_names[variable.group_name].append(variable.name)

        if self.normalize:
            if isinstance(self.normalize, Iterable):
                normalization_variables = set(self.normalize) - self.exclude_normalization
            else:
                normalization_variables = {
                    k
                    for k, v in self._variable_types.items()
                    if v == VariableTypeEnum.CONTINUOUS and k not in self.exclude_normalization
                }

            self.normalizer = self.fit_normalizer(self._dataset_train.select(*normalization_variables))
            self._dataset_train = self.normalizer(self._dataset_train)
            self._dataset_test = self.normalizer(self._dataset_test)
        else:
            self.normalizer = JointTransformModule({})

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
            DataLoader(dataset=self.counterfactuals, collate_fn=identity, batch_size=None),
        ]
        return dataloader_list


class CSuiteDataModule(VariableSpecDataModule):
    """CSuite data module for loading by the datasets name.

    Args:
        dataset_name: Name of the dataset to load.
        batch_size: Batch size for all datasets.
        dataset_path: Path to CSuite dataset mirror.
    """

    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 128,
        dataset_path: str = CAUSICA_DATASETS_PATH,
        load_counterfactual: bool = False,
        load_interventional: bool = False,
        normalize: Union[bool, Iterable[str]] = False,
    ):
        super().__init__(
            root_path=os.path.join(dataset_path, dataset_name),
            batch_size=batch_size,
            dataset_name=dataset_name,
            load_counterfactual=load_counterfactual,
            load_interventional=load_interventional,
            normalize=normalize,
        )


def _check_exists(obj: Any, attribute_name: str):
    """Check if an attribute exists otherwise print a message."""
    try:
        return getattr(obj, attribute_name)
    except AttributeError as exc:
        display_string = attribute_name.replace("_", " ").strip()
        raise ValueError(f"Tried to get {display_string} before data was downloaded.") from exc
