# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch

from ..datasets.dataset import Dataset
from ..datasets.variables import Variables
from ..preprocessing.data_processor import DataProcessor


class IModel(ABC):
    """
    Interface for model:
    create: Create an instance of the concrete class.
    load: Load an instance of the concrete class from a given directory.
    save: Save any data needed to load the model.
    name: Name of objective, to use when finding model to use from string.
    run_train: Train the model.
    impute: Impute missing values:
    """

    def __init__(self, model_id: str, variables: Variables, save_dir: str) -> None:
        """
        Args:
            model_id: Unique model ID for referencing this model instance.
            variables: Information about variables/features used by this model.
            save_dir: Location to save any information about this model, including training data.
                It will be created if it doesn't exist.
        """
        self.model_id = model_id
        self.save_dir = save_dir
        self.variables = variables
        self.data_processor = DataProcessor(variables)

    @classmethod
    @abstractmethod
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
        raise NotImplementedError()

    @classmethod
    @abstractmethod
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
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """
        Name of the model implemented in abstract class.
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self) -> None:
        """
        Save the model.
        """
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()


class IModelForImputation(IModel):
    @abstractmethod
    def impute(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        impute_config_dict: Optional[Dict[str, int]] = None,
        *,
        vamp_prior_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        average: bool = True,
    ) -> np.ndarray:
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
        raise NotImplementedError()


class IBatchImputer(IModelForImputation):
    @abstractmethod
    def impute_processed_batch(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        *,
        sample_count: int,
        preserve_data: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError()


class IModelWithReconstruction(IModel):
    # TODO: At the moment, this is defined in two places: here and in PVAEBaseModel. We should define it in one place instead only
    # TODO: This is different level of abstraction than impute_processed_batch declared above. We should decide which level of abstraction we want
    @abstractmethod
    def reconstruct(
        self,
        data: torch.Tensor,
        mask: Optional[torch.Tensor],
        sample: bool = True,
        count: int = 1,
        **kwargs: Any,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor],]:
        raise NotImplementedError


class IModelForObjective(IBatchImputer, IModelWithReconstruction):
    # TODO this should eventually become an empty class definition that inherits from IBatchImputer and a VAE base model.
    @abstractmethod
    def encode(self, data: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    # Setting evaluation mode
    @abstractmethod
    def set_evaluation_mode(self):
        raise NotImplementedError

    # Gets the device the model is running on
    # TODO: possibly we can get rid of this method. By either:
    # 1) pass device when we construct objective
    # 2) make the code to refer to some static device field (as we probably need to use the same device throughout code)
    # 3) make this class inherit from TorchModel
    @abstractmethod
    def get_device(self):
        raise NotImplementedError

    @abstractmethod
    def get_model_pll(
        self,
        data: np.ndarray,
        feature_mask: np.ndarray,
        target_idx,
        sample_count: int = 50,
    ):
        raise NotImplementedError


class IModelForCausalInference(IModel):
    @abstractmethod
    def get_adj_matrix(self, do_round: bool = True, samples: int = 100, most_likely_graph: bool = False):
        """
        Returns adjacency matrix learned as a numpy array
        """
        raise NotImplementedError


class IModelForInterventions(IModel):
    @abstractmethod
    def sample(
        self,
        Nsamples: int = 100,
        most_likely_graph: bool = False,
        intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        intervention_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):

        """
        Sample from distribution over observations learned by the model. Optionally modify the distribution through interventions.
        """
        raise NotImplementedError

    @abstractmethod
    def log_prob(
        self,
        X: Union[torch.Tensor, np.ndarray],
        intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        intervention_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        conditioning_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        conditioning_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        Nsamples_per_graph: int = 1,
        Ngraphs: Optional[int] = 1000,
        most_likely_graph: bool = False,
        fixed_seed: Optional[int] = None,
    ):
        """
        Evaluate log probability of observations from distribution over observations learned by the model. Optionally modify the distribution through interventions.
        """
        raise NotImplementedError

    @abstractmethod
    def cate(
        self,
        intervention_idxs: Union[torch.Tensor, np.ndarray],
        intervention_values: Union[torch.Tensor, np.ndarray],
        reference_values: Optional[np.ndarray] = None,
        effect_idxs: Optional[np.ndarray] = None,
        conditioning_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        conditioning_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        Nsamples_per_graph: int = 1,
        Ngraphs: Optional[int] = 1000,
        most_likely_graph: bool = False,
        fixed_seed: Optional[int] = None,
    ):
        """
        Evaluate (optionally conditional) average treatment effect given the learnt causal model.
        """
        raise NotImplementedError


class IModelForCounterfactuals(IModel):
    @abstractmethod
    def ite(
        self,
        X: Union[torch.Tensor, np.ndarray],
        intervention_idxs: Union[torch.Tensor, np.ndarray] = None,
        intervention_values: Union[torch.Tensor, np.ndarray] = None,
        reference_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        effect_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        Nsamples_per_graph: int = 1,
        Ngraphs: int = 1000,
        most_likely_graph: bool = False,
        fixed_seed: Optional[int] = None,
    ):
        """
        Calculate the individual treatment effect on interventions on observations X.
        """
        raise NotImplementedError


class IModelForTimeseries(IModel):
    @abstractmethod
    def cate(
        self,
        intervention_idxs: Union[torch.Tensor, np.ndarray],
        intervention_values: Union[torch.Tensor, np.ndarray],
        reference_values: Optional[np.ndarray] = None,
        effect_idxs: Optional[np.ndarray] = None,
        conditioning_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        conditioning_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
        Nsamples_per_graph: int = 1,
        Ngraphs: Optional[int] = 1000,
        most_likely_graph: bool = False,
        fixed_seed: Optional[int] = None,
        conditioning_history: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        """
        Evaluate (optionally conditional) average treatment effect given the learnt causal model.
        """
        raise NotImplementedError
