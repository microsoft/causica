from collections import defaultdict

import pandas as pd
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from causica.datasets.causica_dataset_format import Variable, tensordict_from_variables_metadata
from causica.datasets.normalization import fit_standardizer
from causica.datasets.tensordict_utils import identity, tensordict_shapes
from causica.datasets.variable_types import VariableTypeEnum
from causica.lightning.data_modules.deci_data_module import DECIDataModule


class BasicDECIDataModule(DECIDataModule):
    """A datamodule interface for a dataframe and variable specification."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        variables: list[Variable],
        batch_size: int,
        normalize: bool = False,
        dataset_name: str = "anonymous_dataset",
    ):
        super().__init__()
        self.dataset_df = dataframe
        self.variables = variables
        self._dataset_name = dataset_name
        self.normalize = normalize
        self.batch_size = batch_size

        self._dataset_train = tensordict_from_variables_metadata(self.dataset_df.to_numpy(), self.variables)
        self._variable_shapes = tensordict_shapes(self._dataset_train)
        self._variable_types = {var.group_name: var.type for var in self.variables}
        self._column_names = defaultdict(list)
        for variable in self.variables:
            self._column_names[variable.group_name].append(variable.name)

        if self.normalize:
            continuous_keys = [k for k, v in self._variable_types.items() if v == VariableTypeEnum.CONTINUOUS]
            self.normalizer = fit_standardizer(self._dataset_train.select(*continuous_keys))
            self._dataset_train = self.normalizer(self._dataset_train)

    @property
    def dataset_name(self) -> TensorDict:
        return self._dataset_name

    @property
    def dataset_train(self) -> TensorDict:
        return self._dataset_train

    @property
    def dataset_test(self) -> TensorDict:
        raise RuntimeError("Not defined")

    @property
    def variable_shapes(self) -> dict[str, torch.Size]:
        """Get the shape of each variable in the dataset."""
        return self._variable_shapes

    @property
    def variable_types(self) -> dict[str, VariableTypeEnum]:
        """Get the type of each variable in the dataset."""
        return self._variable_types

    @property
    def column_names(self) -> dict[str, list[str]]:
        """Get a map of the variable names and the corresponding columns of the original dataset."""
        return self._column_names

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            collate_fn=identity,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
