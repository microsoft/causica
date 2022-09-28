from __future__ import annotations

import copy
import logging
import os
from time import gmtime, strftime
from typing import Any, Dict, Tuple, Type, TypeVar, Union

import torch

from ..datasets.variables import Variables
from ..utils.helper_functions import maintain_random_state
from ..utils.io_utils import read_json_as, save_json, save_txt
from ..utils.torch_utils import get_torch_device, set_random_seeds
from .model import Model
from .torch_training_types import LossConfig, LossResults

logger = logging.getLogger(__name__)

# Create type variable with upper bound of `TorchModel`, in order to precisely specify return types of create/load
# methods as the subclass on which they are called
T = TypeVar("T", bound="TorchModel")


class ONNXNotImplemented(NotImplementedError):
    pass


class TorchModel(Model, torch.nn.Module):
    # TODO all but model_file can go in Model
    _model_config_path = "model_config.json"
    _model_type_path = "model_type.txt"
    _variables_path = "variables.json"
    model_file = "model.pt"
    best_model_file = "best_model.pt"

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        **_,
    ) -> None:
        """
        Args:
            model_id: Unique model ID for referencing this model instance.
            variables: Information about variables/features used by this model.
            save_dir: Location to save any information about this model, including training data.
                It will be created if it doesn't exist.
            device: Name of Torch device to create the model on. Valid options are 'cpu', 'gpu', or a device ID
                (e.g. 0 or 1 on a two-GPU machine).
        """
        torch.nn.Module.__init__(self)
        Model.__init__(self, model_id, variables, save_dir)
        self.device = device

    def save(self, best: bool = False) -> None:
        """
        Save the torch model state_dict, as well as an ONNX representation of the model,
        if implemented.

        Args:
            best: Flag indicating whether this is a new best checkpoint. This saves to a different location.
        """
        self.variables.save(os.path.join(self.save_dir, self._variables_path))
        if best:
            model_path = os.path.join(self.save_dir, self.best_model_file)
        else:
            model_path = os.path.join(self.save_dir, self.model_file)

        torch.save(self.state_dict(), model_path)

        # TODO save variables? For most cases, the 'create' method will already
        # have saved variables.

        # Save ONNX files for all model components.
        # Generating mock ONNX input will affect random seed.
        # So store and restore.
        with maintain_random_state():
            try:
                self.save_onnx(self.save_dir)
            except ONNXNotImplemented:
                logger.debug("Save ONNX not implemented for this model.")

    def save_onnx(self, save_dir: str, best: bool = False) -> None:
        raise ONNXNotImplemented

    def validate_loss_config(self, loss_config: LossConfig):
        # Implement in child classes to check the loss config is valid
        # for this specific model, if you want to use the torch_training.train_model function.
        raise NotImplementedError

    def loss(self, loss_config: LossConfig, input_tensors: Tuple[torch.Tensor]) -> LossResults:
        # Implement this if you want to be able to use torch_training.train_model function.
        raise NotImplementedError

    @classmethod
    def load(
        cls: Type[T],
        model_id: str,
        save_dir: str,
        device: Union[str, int, torch.device],
    ) -> T:
        """
        Load an instance of a model.

        Args:
            model_id: Unique model ID for referencing this model instance.
            save_dir: Save directory for this model.
            device: Name of Torch device to create the model on. Valid options are 'cpu', 'gpu', or a
                device ID (e.g. 0 or 1 on a two-GPU machine). Can also pass a torch.device directly.

        Returns:
            Instance of concrete implementation of `TorchModel` class.
        """
        # Load variables.
        variables_path = os.path.join(save_dir, cls._variables_path)
        variables = Variables.create_from_json(variables_path)

        # Load model config.
        model_config_path = os.path.join(save_dir, cls._model_config_path)
        model_config_dict = read_json_as(model_config_path, dict)
        model_config_dict = _set_random_seed_and_remove_from_config(model_config_dict)

        # Finally, get a model instance and allow that model to load anything else it needs.
        model = cls._load(model_id, variables, save_dir, device, **model_config_dict)
        return model

    def reload_saved_parameters(self):
        # Used to implement 'rewind_to_best_epoch' behaviour.
        self.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_file)))

    @classmethod
    def create(
        cls: Type[T],
        model_id: str,
        save_dir: str,
        variables: Variables,
        model_config_dict: Dict[str, Any],
        device: Union[str, int, torch.device],
    ) -> T:
        """
        TODO most of this can go in Model class
        Create a new instance of a model with given type.

        Args:
            model_id: Unique model ID for referencing this model instance.
            save_dir: Location to save all model information to.
            variables: Information about variables/features used
                by this model.
            model_config_dict: Any other parameters needed by a specific concrete class. Of
                the form {arg_name: arg_value}. e.g. {"embedding_dim": 10, "latent_dim": 20}
            device: Name of device to load the model on. Valid options are 'cpu', 'gpu', or a device ID
                (e.g. 0 or 1 on a two-GPU machine). Can also pass a torch.device directly. Ignored for some models.

        Returns:
            model: Instance of concrete implementation of `Model` class.
        """
        # TODO set random seed before calling _create? Many subclass implementations of _create
        # include setting random seed.

        torch_device = get_torch_device(device)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save all the model information.
        # Save model config to save dir.
        model_config_save_path = os.path.join(save_dir, cls._model_config_path)
        save_json(model_config_dict, model_config_save_path)

        model_config_dict = _set_random_seed_and_remove_from_config(model_config_dict)
        model = cls._create(model_id, variables, save_dir, device=torch_device, **model_config_dict)

        # Save variables file.
        variables_path = os.path.join(save_dir, cls._variables_path)
        variables.save(variables_path)

        # Save model type.
        model_type_path = os.path.join(save_dir, cls._model_type_path)
        save_txt(cls.name(), model_type_path)

        # Save the model that has just been created.
        model.save()
        return model

    @classmethod
    def _create(
        cls: Type[T],
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        **model_config_dict,
    ) -> T:
        model = cls(model_id, variables, save_dir, device, **model_config_dict)
        num_trainable_parameters = sum(p.numel() for p in model.parameters())
        logger.info(f"Model has {num_trainable_parameters} trainable parameters.")
        return model

    @classmethod
    def _load(
        cls: Type[T],
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: Union[str, int, torch.device],
        **model_config_dict,
    ) -> T:
        """
        Load a saved instance.

        Args:
            model_id: Unique model ID for referencing this model instance.
            variables (Variables): Information about variables/features used
                by this model.
            save_dir: Location to load model from.
            device: Name of Torch device to create the model on. Valid options are 'cpu', 'gpu',
                or a device ID (e.g. 0 or 1 on a two-GPU machine). Can also pass a torch.device directly.
            random_seed: Random seed to set before creating model. Defaults to 0.
            **model_config_dict: Any other arguments needed by the concrete class. Defaults can be specified in the
                concrete class. e.g. ..., embedding_dim, latent_dim=20, ...

        Returns:
            Instance of concrete implementation of TorchModel class.
        """
        torch_device = get_torch_device(device)
        model = cls._create(model_id, variables, save_dir, torch_device, **model_config_dict)
        model_path = os.path.join(save_dir, cls.model_file)
        model.load_state_dict(torch.load(model_path, map_location=torch_device))
        return model

    def get_device(self) -> torch.device:
        return self.device

    def set_evaluation_mode(self):
        self.eval()

    def _create_train_output_dir_and_save_config(self, train_config_dict: dict) -> str:
        """

        Args:
            train_config_dict (dict): hyperparameters for training

        Returns:
            str: path to the newly created training directory
        """
        # Create directory and save config to that folder.
        starttime = strftime("%Y-%m-%d_%H%M%S", gmtime())
        train_output_dir = os.path.join(self.save_dir, f"train_{starttime}")
        os.makedirs(train_output_dir, exist_ok=True)

        train_config_save_path = os.path.join(train_output_dir, "train_config.json")
        save_json(train_config_dict, train_config_save_path)
        return train_output_dir


def _set_random_seed_and_remove_from_config(model_config_dict: Dict) -> Dict:
    # Set random seed to model_config_dict.get('random_seed', 0)
    # If 'random_seed' is in model_config_dict, create a copy of model_config_dict that has random_seed removed.
    if "random_seed" in model_config_dict:
        random_seed = model_config_dict["random_seed"]
        model_config_dict = copy.copy(model_config_dict)
        del model_config_dict["random_seed"]
    else:
        random_seed = 0
    set_random_seeds(random_seed)
    return model_config_dict
