import abc

import pytorch_lightning as pl
import torch
from tensordict import TensorDict

from causica.datasets.causica_dataset_format import CounterfactualWithEffects, InterventionWithEffects


class VariableSpecModule(pl.LightningModule, abc.ABC):
    """
    An Abstract Base Class for `LightningModule`s that use the VariableSpecDataModule.

    This class provides a default implementation of the `test_step` method that
    dispatches to the separate test steps for each dataloader.
    """

    def test_step(self, *args, **kwargs):
        """
        Dispatch to the appropriate test step based on the dataloader index.

        The dataloader indexing is handled by Lightning and is based on the order
        of the dataloaders in the `test_dataloader` method.

        This assumes that the dataloaders are in the following order:
        Test data
        Graph data
        Intervention data
        Counterfactual data

        See the superclass for *args and **kwargs conventions.
        """
        dataloader_idx = args[2]
        if dataloader_idx == 0:
            return self.test_step_observational(*args, **kwargs)
        elif dataloader_idx == 1:
            return self.test_step_graph(*args, **kwargs)
        elif dataloader_idx == 2:
            return self.test_step_interventions(*args, **kwargs)
        elif dataloader_idx == 3:
            return self.test_step_counterfactuals(*args, **kwargs)

    @abc.abstractmethod
    def test_step_observational(self, batch: TensorDict, *args, **kwargs):
        """Test step method for the test data."""

    @abc.abstractmethod
    def test_step_graph(self, true_adj_matrix: torch.Tensor, *args, **kwargs):
        """Test step method for the true adjacency matrix."""

    @abc.abstractmethod
    def test_step_interventions(self, interventions: InterventionWithEffects, *args, **kwargs):
        """Test step method for the interventions."""

    @abc.abstractmethod
    def test_step_counterfactuals(self, counterfactuals: CounterfactualWithEffects, *args, **kwargs):
        """Test step method for the counterfactuals."""
