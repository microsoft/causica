# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

import numpy as np
from lingam import VARLiNGAM as varlingam_alg

from ..datasets.dataset import Dataset, TemporalDataset
from ..datasets.variables import Variables
from ..models.imodel import IModelForCausalInference
from ..models.model import Model
from ..utils.io_utils import read_json_as, save_json, save_txt

T = TypeVar("T", bound="VARLiNGAM")

logger = logging.getLogger(__name__)


class VARLiNGAM(Model, IModelForCausalInference):
    _model_config_path = "model_config.json"
    _model_type_path = "model_type.txt"
    _variables_path = "variables.json"
    model_file = "model.pkl"

    def __init__(self, model_id: str, variables: Variables, save_dir: str, **model_config_dict):
        super().__init__(model_id, variables, save_dir)
        lag_order = model_config_dict.get("lag_order", 1)
        prune = model_config_dict.get("prune", False)
        random_state = model_config_dict.get("random_seed", 0)
        criterion = model_config_dict.get("criterion", None)
        self.learner = varlingam_alg(lags=lag_order, prune=prune, random_state=random_state, criterion=criterion)
        self.model_config_dict = model_config_dict

    @classmethod
    def name(cls) -> str:
        return "varlingam"

    @classmethod
    def create(
        cls: Type[T],
        model_id: str,
        save_dir: str,
        variables: Variables,
        model_config_dict: Dict[str, Any],
        device: Union[str, int],
    ) -> T:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model config
        model_config_save_path = os.path.join(save_dir, cls._model_config_path)
        save_json(model_config_dict, model_config_save_path)

        # Save variables file.
        variables_path = os.path.join(save_dir, cls._variables_path)
        variables.save(variables_path)

        # Save model type.
        model_type_path = os.path.join(save_dir, cls._model_type_path)
        save_txt(cls.name(), model_type_path)

        return cls(model_id=model_id, variables=variables, save_dir=save_dir, **model_config_dict)

    def get_adj_matrix(self):
        # This returns an adjacency matrix in the form [lag, from, to]
        # Since the original adjacency matrix is [lag,to,from], transpose it to [lag, from,to]
        # This contains an array of True/False
        return np.transpose(np.abs(self.learner.adjacency_matrices_) > 0, axes=[0, 2, 1])

    def get_transition_matrix(self):
        # This return the transition matrix in the form [lag, from, to]
        return np.transpose(self.learner.adjacency_matrices_, axes=[0, 2, 1])

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Optional[Dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        # Get data

        data, _ = dataset.train_data_and_mask

        # Extract data and fit on longest series.
        assert isinstance(dataset, TemporalDataset)
        assert dataset.train_segmentation is not None
        longest_idx = np.argmax([seg[1] - seg[0] for seg in dataset.train_segmentation])
        longest_series = data[dataset.train_segmentation[longest_idx][0] : dataset.train_segmentation[longest_idx][1]]
        # Training
        self.learner.fit(longest_series)
        # Save model
        self.save()

    def save(self) -> None:
        # Save variables
        os.makedirs(self.save_dir, exist_ok=True)
        self.variables.save(os.path.join(self.save_dir, self._variables_path))
        # Save model in pickle format
        # Cannot just save the adjacency matrix, since the VARLiNGaM learner class also has lags, residuals and causal_order properties.
        # It is easier to just store the entire model as .pkl
        model_path = os.path.join(self.save_dir, self.model_file)
        logger.info(f"saved model to {model_path}")
        with open(f"{model_path}", "wb") as f:
            pickle.dump(self.learner, f)

    @classmethod
    def load(cls, model_id: str, save_dir: str, device: Union[str, int]) -> "VARLiNGAM":
        # Load learner
        model_path = os.path.join(save_dir, cls.model_file)
        with open(f"{model_path}", "rb") as f:
            learner = pickle.load(f)

        # Load variables.
        variables_path = os.path.join(save_dir, cls._variables_path)
        variables = Variables.create_from_json(variables_path)

        # Load model config.
        model_config_path = os.path.join(save_dir, cls._model_config_path)
        model_config_dict = read_json_as(model_config_path, dict)

        model = cls.create(model_id, save_dir, variables, model_config_dict, device)
        model.learner = learner
        return model
