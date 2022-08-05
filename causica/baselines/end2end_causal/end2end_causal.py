from __future__ import annotations

import copy
import logging
import os
from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch

from ...datasets.dataset import Dataset
from ...datasets.variables import Variables
from ...models.imodel import IModel, IModelForCausalInference, IModelForInterventions
from ...models.model import Model
from ...models.torch_model import _set_random_seed_and_remove_from_config
from ...utils.io_utils import save_json, save_txt
from ...utils.torch_utils import get_torch_device

T = TypeVar("T", bound="End2endCausal")


logger = logging.getLogger(__name__)


class End2endCausal(Model, IModelForCausalInference, IModelForInterventions):
    _discovery_model_save_dir = "discovery_model"
    _inference_model_save_dir = "inference_model"

    _model_config_path = "model_config.json"
    _model_type_path = "model_type.txt"
    _variables_path = "variables.json"
    model_file = "model.pt"

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        discovery_model: Optional[IModelForCausalInference],
        inference_model: IModelForInterventions,
        **model_config_dict,
    ):

        """
        Wrapper class for a causal discovery model and a separate causal inference model.

        Args:
            model_id: Unique model ID for referencing this model instance.
            variables: Information about variables/features used by this model.
            save_dir: Location to save any information about this model, including training data. This wrapper is stateless but this information is needed for test result saving purposes.
        """
        super().__init__(model_id, variables, save_dir)
        _ = device
        self.discovery_model = discovery_model
        self.inference_model = inference_model
        self.model_config_dict = model_config_dict

        self.inference_model.save_dir = os.path.join(save_dir, self._inference_model_save_dir)

    @abstractmethod
    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Optional[Dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        pass

    def sample(
        self,
        Nsamples: int = 100,
        most_likely_graph: bool = False,
        intervention_idxs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        intervention_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> torch.Tensor:
        """
        Sample from distribution over observations learned by the model. Optionally modify the distribution through interventions.
        """
        return self.inference_model.sample(
            Nsamples=Nsamples,
            most_likely_graph=most_likely_graph,
            intervention_idxs=intervention_idxs,
            intervention_values=intervention_values,
        )

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
        return self.inference_model.log_prob(
            X=X,
            Nsamples_per_graph=Nsamples_per_graph,
            most_likely_graph=most_likely_graph,
            intervention_idxs=intervention_idxs,
            intervention_values=intervention_values,
        )

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
        return self.inference_model.cate(
            intervention_idxs=intervention_idxs,
            intervention_values=intervention_values,
            reference_values=reference_values,
            effect_idxs=effect_idxs,
            conditioning_idxs=conditioning_idxs,
            conditioning_values=conditioning_values,
            Nsamples_per_graph=Nsamples_per_graph,
            Ngraphs=Ngraphs,
            most_likely_graph=most_likely_graph,
        )

    def get_adj_matrix(self, do_round: bool = True, samples: int = 100, most_likely_graph: bool = False):
        """
        Returns adjacency matrix learned as a numpy array
        """
        _ = most_likely_graph
        assert self.discovery_model is not None
        return self.discovery_model.get_adj_matrix(do_round=do_round, samples=samples)

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        raise NotImplementedError

    @staticmethod
    def split_configs(config_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        config_dict = copy.deepcopy(config_dict)
        discovery_config = config_dict.pop("discovery_config", {})
        inference_config = config_dict.pop("inference_config", {})

        for key, val in config_dict.items():
            key_split = key.split("_", 1)
            if len(key_split) != 2:
                key_type = None
            else:
                key_type, _ = key_split

            if key_type in ["discovery", "inference"]:
                raise ValueError(
                    "Config parsing has been changed. You need to specify discovery and inference "
                    "configs in nested dicts with keys `discovery_config` and `inference_config`."
                )
            else:
                # Put in both.
                discovery_config[key] = val
                inference_config[key] = val

        logger.info(f"Using discovery config:\n{discovery_config} and inference config:\n{inference_config}")

        return discovery_config, inference_config

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
        os.mkdir(os.path.join(save_dir, cls._discovery_model_save_dir))
        os.mkdir(os.path.join(save_dir, cls._inference_model_save_dir))
        base_model_save_path = os.path.join(save_dir, cls._model_config_path)
        discovery_model_save_path = os.path.join(save_dir, cls._discovery_model_save_dir, cls._model_config_path)
        inference_model_save_path = os.path.join(save_dir, cls._inference_model_save_dir, cls._model_config_path)
        discovery_config, inference_config = cls.split_configs(model_config_dict)

        save_json(model_config_dict, base_model_save_path)
        save_json(discovery_config, discovery_model_save_path)
        save_json(inference_config, inference_model_save_path)

        # Save variables file.
        variables_path = os.path.join(save_dir, cls._variables_path)
        variables.save(variables_path)

        # Save model type.
        model_type_path = os.path.join(save_dir, cls._model_type_path)
        save_txt(cls.name(), model_type_path)  # type: ignore

        # Mock model file (needed for correct results aggregation)
        # TODO: add proper serialization (that would come for free if we inherit from TorchModel)
        # Alternatively, change assumption in the pipeline, not to look for model.pt, but other file
        model_path = os.path.join(save_dir, cls.model_file)
        with open(model_path, "w", encoding="utf-8") as file:
            file.write("")

        model_config_dict = _set_random_seed_and_remove_from_config(model_config_dict)

        torch_device = get_torch_device(device)

        return cls(model_id=model_id, variables=variables, save_dir=save_dir, device=torch_device, **model_config_dict)

    def save(self) -> None:
        """
        Save the model.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()
