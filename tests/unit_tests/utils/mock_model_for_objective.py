from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from scipy.sparse import issparse

from causica.datasets.dataset import Dataset
from causica.datasets.variables import Variables
from causica.models.imodel import IModel, IModelForObjective


# pylint: disable=unused-argument
class MockModelForObjective(IModelForObjective):
    @classmethod
    def create(
        cls,
        model_id: str,
        save_dir: str,
        variables: Variables,
        model_config_dict: Dict[str, Any],
        device: Union[str, int],
    ) -> IModel:
        """
        Create a new instance of a model with given type.

        Args:
            model_id (str): Unique model ID for referencing this model instance.
            save_dir (str): Location to save all model information to.
            device (str or int): Name of device to load the model on. Valid options are 'cpu', 'gpu', or a device ID
                (e.g. 0 or 1 on a two-GPU machine).
            variables (Variables): Information about variables/features used
                by this model.
            model_config_dict (dictionary): Any other parameters needed by a specific concrete class. Of
                the form {arg_name: arg_value}. e.g. {"embedding_dim": 10, "latent_dim": 20}

        Returns:
            model: Instance of concrete implementation of `Model` class.
        """
        _, _ = model_config_dict, device
        return cls(model_id, variables, save_dir)

    @classmethod
    def load(cls, model_id: str, save_dir: str, device: Union[str, int]) -> IModel:
        """
        Load an instance of a model.

        Args:
            model_id (str): Unique model ID for referencing this model instance.
            save_dir (str): Save directory for this model.
            device (str or int): Name of device to load the model on. Valid options are 'cpu', 'gpu', or a device ID
                (e.g. 0 or 1 on a two-GPU machine).
            variables (Variables): Information about variables/features used
                by this model.

        Returns:
            Instance of concrete implementation of `Model` class.
        """
        pass

    @classmethod
    def name(cls) -> str:
        """
        Name of the model implemented in abstract class.
        """
        return "mock_model_for_objective"

    def save(self) -> None:
        """
        Save the model.
        """
        pass

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Optional[Dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """
        Train the model.
        Training results will be saved.

        Args:
            dataset: Dataset object with data and masks in unprocessed form.
            train_config_dict (dictionary): Any other parameters needed by a specific concrete class. Of
                the form {arg_name: arg_value}. e.g. {"learning_rate": 1e-3, "epochs": 100}
            report_progress_callback: Function to report model progress for API.
        """
        pass

    def impute(
        self,
        data,
        mask,
        impute_config_dict=None,
        vamp_prior_data=None,
        average=None,
    ):
        """
        Fill in unobserved variables using a trained model.
        Data should be provided in unprocessed form, and will be processed before running, and
        will be de-processed before returning (i.e. variables will be in their normal, rather than
        squashed, ranges).

        Args:
            data (numpy array of shape (batch_size, feature_count)): Data to be used to train the model,
                in unprocessed form.
            mask (numpy array of shape (batch_size, feature_count)): Corresponding mask, where observed
                values are 1 and unobserved values are 0.
            impute_config_dict (dictionary): Any other parameters needed by a specific concrete class. Of
                the form {arg_name: arg_value}. e.g. {"sample_count": 10}
            vamp_prior_data (Tuple of (data, mask)): Data to be used to fill variables if using the vamp
                prior method. This defaults to None, in which case the vamp prior method will not be used.
            average (bool): Whether or not to return the averaged results accross Monte Carlo samples, Defaults to True.

        Returns:
            imputed (numpy array of shape (batch_size, feature_count)): Input data with missing values filled in.
        """
        if issparse(data):
            data = data.toarray()
        return np.ones_like(data)

    def impute_processed_batch(
        self, data: torch.Tensor, mask: torch.Tensor, sample_count: int, preserve_data: bool = True, **kwargs
    ) -> torch.Tensor:
        return torch.ones_like(data).unsqueeze(0).repeat(sample_count, 1, 1)

    def encode(self, data: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_dim = 10
        batch_size = data.shape[0]
        # encoded = torch.ones(batch_size, latent_dim)
        encoded = torch.rand(batch_size, latent_dim)
        return (encoded, encoded)

    def reconstruct(
        self, data: torch.Tensor, mask: Optional[torch.Tensor], sample: bool = True, count: int = 1, **kwargs: Any
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor],]:
        latent_dim = 10
        batch_size = data.shape[0]
        encoded = torch.ones(batch_size, latent_dim)
        decoded = torch.ones_like(data)
        return (decoded, decoded), encoded, (encoded, encoded)

    def set_evaluation_mode(self):
        pass

    def get_device(self):
        return torch.device("cpu")

    def get_model_pll(
        self,
        data: np.ndarray,
        feature_mask: np.ndarray,
        target_idx,
        sample_count: int = 50,
    ):
        pass
