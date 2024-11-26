"""Lightning Classes for loading data in the default format used in Azure Blob Storage."""
import functools
import os
from collections import defaultdict
from functools import partial
from typing import Any, Iterable, Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase
from torch.utils.data import DataLoader

from causica.datasets.causica_dataset_format import CAUSICA_DATASETS_PATH, DataEnum, VariablesMetadata, load_data
from causica.datasets.interventional_data import CounterfactualData, InterventionData
from causica.datasets.normalization import (
    FitNormalizerType,
    Normalizer,
    chain_normalizers,
    fit_log_normalizer,
    fit_standardizer,
)
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
        standardize: Union[bool, Iterable[str]] = False,
        log_normalize: Union[bool, Iterable[str]] = False,
        exclude_standardization: Iterable[str] = tuple(),
        exclude_log_normalization: Iterable[str] = tuple(),
        default_offset: float = 1.0,
        log_normalize_min_margin: float = 0.0,
        fit_normalizer_on_test_sets: bool = False,
        load_counterfactual: bool = False,
        load_interventional: bool = False,
        load_validation: bool = False,
        **storage_options: Any,
    ):
        """
        Args:
            root_path: Path to directory with causal data
            batch_size: Batch size for training and test data.
            dataset_name: A name for the dataset
            standardize: Whether to standardize the data or not. It is applied to all continuous variables if True, or applied to those
                variables specifed in normalize except those specified in
                `exclude_normalization`. The standardizer is column-wise: (x_i-mean_i)/std_i for ith column.
                If both standardize and log_normalize are True, log_normalize will be applied first.
            log_normalize: Whether to log normalize the data. If True, it will log normalize all continuous variables.
                Or it will be applied to those variables specified in log_normalize except those specified in `exclude_log_normalization`.
                The operation is
                log(x_i - min_i * (min_i < 0) + min_i + (max_i - min_i) * min_margin + offset). Also see the reference
                in datasets.normalization.LogTransform. If both standardize and log_normalize are True,
                log_normalize will be applied first.
            exclude_standardization: Which variables to exclude from standardization
            exclude_log_normalization: Which variables to exclude from log normalization
            default_offset: Default offset for log normalization.
            log_normalize_min_margin: Minimum margin for log normalization.
            fit_normalizer_on_test_sets: Whether to normalize all data. If True, normalization will be fitted on all data splits.
            load_counterfactual: Whether counterfactual data should be loaded
            load_interventional: Whether interventional data should be loaded
            load_validation: Whether to load the validation dataset
            **storage_options: Storage options forwarded to `fsspec` when loading files.
        """
        super().__init__()
        self.batch_size = batch_size
        self._dataset_name = dataset_name
        self.root_path = root_path
        self.batch_size = batch_size
        self.storage_options = storage_options
        self.standardize = standardize
        self.log_normalize = log_normalize
        self.exclude_standardization = set(exclude_standardization)
        self.exclude_log_normalization = set(exclude_log_normalization)
        self.load_counterfactual = load_counterfactual
        self.load_interventional = load_interventional
        self.load_validation = load_validation
        self.default_offset = default_offset
        self.log_normalize_min_margin = log_normalize_min_margin
        self.fit_normalizer_on_test_sets = fit_normalizer_on_test_sets

        self.use_normalizer = standardize or log_normalize
        self.normalizer: Optional[Normalizer] = None
        self._dataset_train: TensorDictBase
        self._dataset_test: TensorDictBase
        self._dataset_valid: TensorDictBase
        self.true_adj: torch.Tensor
        self.save_hyperparameters()

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
    def dataset_valid(self) -> TensorDict:
        return _check_exists(self, "_dataset_valid")

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def create_normalizer(self, normalization_variables: set[str]) -> FitNormalizerType:
        """Return a fitting method for a sequence of normalizers.

        This function is used to return a fitting method for a sequence of normalizers (e.g. log_normalize and standardize).
        The variables for each normalizer is normalization_variables - exclude_normalizer, where normalization_variables is
        a large set of variables (e.g. all continuous variables) and exclude_normalizer is a set of variables that should be excluded specific
        to this normalizer (e.g. exclude_log_normalization and exclude_standardization).

        Args:
            normalization_variables: A larger set of variables to be normalized. It should at least contain the union of variables from all normalizers.

        Returns:
            Return a fitting method for a sequence of normalizers.
        """
        preprocessing: list[FitNormalizerType] = []
        # Setup different separate normalizers
        if self.log_normalize:
            log_normalize_keys = normalization_variables - self.exclude_log_normalization
            if isinstance(self.log_normalize, Iterable):
                log_normalize_keys = set(self.log_normalize) - self.exclude_log_normalization

            fit_log_normalizer_with_key = functools.partial(
                fit_log_normalizer,
                default_offset=self.default_offset,
                min_margin=self.log_normalize_min_margin,
                keys=list(log_normalize_keys),
            )
            preprocessing.append(fit_log_normalizer_with_key)

        if self.standardize:
            standardize_keys = normalization_variables - self.exclude_standardization
            if isinstance(self.standardize, Iterable):
                standardize_keys = set(self.standardize) - self.exclude_standardization

            fit_standardizer_with_key = functools.partial(
                fit_standardizer,
                keys=list(standardize_keys),
            )
            preprocessing.append(fit_standardizer_with_key)

        return chain_normalizers(*preprocessing)

    def _load_all_data(self, variables_metadata: VariablesMetadata):
        _load_data = partial(
            load_data, root_path=self.root_path, variables_metadata=variables_metadata, **self.storage_options
        )

        dataset_train = _load_data(data_enum=DataEnum.TRAIN)
        dataset_test = _load_data(data_enum=DataEnum.TEST)
        if self.load_validation:
            dataset_valid = _load_data(data_enum=DataEnum.VALIDATION)
            assert isinstance(dataset_valid, TensorDict)
            self._dataset_valid = dataset_valid
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

        if self.use_normalizer:
            # Only applied to continuous variables
            normalization_variables = {k for k, v in self._variable_types.items() if v == VariableTypeEnum.CONTINUOUS}
            normalization_sets = [self._dataset_train, self._dataset_test]
            if self.load_validation:
                normalization_sets.append(self._dataset_valid)
            normalization_data = (
                torch.cat(normalization_sets) if self.fit_normalizer_on_test_sets else self._dataset_train
            )
            self.normalizer = self.create_normalizer(normalization_variables)(
                normalization_data.select(*normalization_variables)
            )
            self.normalize_data()
        else:
            self.normalizer = JointTransformModule({})

    def normalize_data(self):
        self._dataset_train = self.normalizer(self._dataset_train)
        self._dataset_test = self.normalizer(self._dataset_test)
        if self._dataset_test.apply(torch.isnan).any():
            raise ValueError(
                "NaN values found in the test data after normalization. Consider changing the normalization parameters."
            )
        if self.load_validation:
            self._dataset_valid = self.normalizer(self._dataset_valid)
            if self._dataset_valid.apply(torch.isnan).any():
                raise ValueError(
                    "NaN values found in the validation data after normalization. Consider changing the normalization parameters."
                )

        if self.load_interventional:
            for intervention in self.interventions:
                for i in intervention:
                    if isinstance(i, InterventionData):
                        i.intervention_data = self.normalizer(i.intervention_data)
                        i.intervention_values = self.normalizer(i.intervention_values)
                        i.condition_values = self.normalizer(i.condition_values)

        if self.load_counterfactual:
            for cf in self.counterfactuals:
                for c in cf:
                    if isinstance(c, CounterfactualData):
                        c.counterfactual_data = self.normalizer(c.counterfactual_data)
                        c.factual_data = self.normalizer(c.factual_data)
                        c.intervention_values = self.normalizer(c.intervention_values)

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
        standardize: Union[bool, Iterable[str]] = False,
    ):
        super().__init__(
            root_path=os.path.join(dataset_path, dataset_name),
            batch_size=batch_size,
            dataset_name=dataset_name,
            load_counterfactual=load_counterfactual,
            load_interventional=load_interventional,
            standardize=standardize,
        )


def _check_exists(obj: Any, attribute_name: str):
    """Check if an attribute exists otherwise print a message."""
    try:
        return getattr(obj, attribute_name)
    except AttributeError as exc:
        display_string = attribute_name.replace("_", " ").strip()
        raise ValueError(f"Tried to get {display_string} before data was downloaded.") from exc
