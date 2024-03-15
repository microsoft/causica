from collections import defaultdict
from collections.abc import Iterable

import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from scotch.scotch_utils.scotch_utils import check_temporal_tensordict_shapes, temporal_tensordict_shapes
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import DataLoader
from torchsde import BrownianInterval

from causica.datasets.causica_dataset_format import VariablesMetadata
from causica.datasets.tensordict_utils import identity
from causica.datasets.variable_types import VariableTypeEnum
from causica.lightning.data_modules.deci_data_module import DECIDataModule


class SCOTCHDataModule(DECIDataModule):
    """A SCOTCH datamodule interface for a dataframe and variable specification."""

    def __init__(
        self,
        ts: list[int],
        training_data: TensorDict,
        validation_data: TensorDict,
        true_graph: Tensor,
        variables_metadata: VariablesMetadata,
        batch_size: int,
        validation_bm: BrownianInterval = None,
        dataset_name: str = "anonymous_dataset",
    ):
        """Constructor for SCOTCHDataModule, given a dataframe containing the temporal data, and variable specification.

        At present it is assumed that all trajectories in the dataset are observed at the same time points; however,
        in future it is planned to extend to heterogenous time points (which will be encoded as separate rows in the
        dataframe, with a column indicating the time point for each row).

        Args:
            ts (list[int]): An iterable containing the time points; assumed to be the same for each observation.
            training_data (TensorDict): A TensorDict containing the temporal data. In particular, each entry is of shape
                (num_observations, num_time_points, num_variables_in_group), and the keys are the variable group names.
            validation_data (TensorDict): Like training_data, but used for validation.
            validation_bm (BrownianInterval): Brownian motion from the SDE trajectories in the validation data; used
                to fix the stochasticity in SDE for computing test MSE. Should have size (val_size, state_size)
                If not available (for non synthetic data), then default to None.
            true_graph (Tensor): The true graph adjacency matrix, of shape (state_size, state_size).
            others: See BasicDECIDataModule.
        """
        super().__init__()
        self._dataset_name = dataset_name
        self.batch_size = batch_size

        self._true_graph = true_graph

        self._dataset_train = training_data
        self._dataset_val = validation_data

        # Check that shapes are valid
        assert check_temporal_tensordict_shapes(len(ts), self._dataset_train)
        self._variable_shapes = temporal_tensordict_shapes(self._dataset_train)
        self._variable_types = {var.group_name: var.type for var in variables_metadata.variables}
        self._column_names = defaultdict(list)
        for variable in variables_metadata.variables:
            self._column_names[variable.group_name].append(variable.name)

        # normalization removed (see BasicDECIDataModule)

        self._ts = ts
        self._bm_val = validation_bm

    @property
    def ts(self) -> Iterable[int]:
        return self._ts

    @property
    def bm_validation(self) -> BrownianInterval:
        return self._bm_val

    @property
    def true_graph(self) -> Tensor:
        return self._true_graph

    @property
    def dataset_name(self) -> TensorDict:
        return self._dataset_name

    @property
    def dataset_train(self) -> TensorDict:
        return self._dataset_train

    @property
    def dataset_val(self) -> TensorDict:
        return self._dataset_val

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
            drop_last=False,  # allow for incomplete batches
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.dataset_val,
            collate_fn=identity,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,  # allow for incomplete batches
        )
