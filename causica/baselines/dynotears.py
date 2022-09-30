from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

import networkx as nx
import numpy as np
import pandas as pd
from causalnex.structure import StructureModel
from causalnex.structure.dynotears import from_pandas_dynamic

from ..datasets.dataset import Dataset, TemporalDataset
from ..datasets.variables import Variables
from ..models.imodel import IModelForCausalInference
from ..models.model import Model
from ..models.torch_model import _set_random_seed_and_remove_from_config
from ..utils.io_utils import read_json_as, save_json, save_txt

T = TypeVar("T", bound="Dynotears")

logger = logging.getLogger(__name__)


class Dynotears(Model, IModelForCausalInference):
    _model_config_path = "model_config.json"
    _model_type_path = "model_type.txt"
    _variables_path = "variables.json"
    model_file = "model.pkl"

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        lag: int,
        lambda_w: float = 0.1,
        lambda_a: float = 0.1,
    ):
        """
        Init method for dynotears instance.
        Args:
            lag: the model lag.
            lambda_w: The l1 sparse regularization of instantaneous weighted adj matrix.
            lambda_a: The l1 sparse regularization of lagged weighted adj matrix.
        """
        super().__init__(model_id, variables, save_dir)
        self.lag = lag
        self.lambda_w = lambda_w
        self.lambda_a = lambda_a
        self.learner = None

    @classmethod
    def name(cls) -> str:
        return "dynotears"

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

        # set seed and remove from config
        model_config_dict = _set_random_seed_and_remove_from_config(model_config_dict)

        # Save variables file.
        variables_path = os.path.join(save_dir, cls._variables_path)
        variables.save(variables_path)

        # Save model type.
        model_type_path = os.path.join(save_dir, cls._model_type_path)
        save_txt(cls.name(), model_type_path)

        return cls(model_id=model_id, variables=variables, save_dir=save_dir, **model_config_dict)

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Optional[Dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """
        This runs the training algorithm of dynotears and assigned to self.learner
        Args:
            dataset: The temporal dataset containing the data.
            train_config_dict: The train config dict, containing the following: max_iter, h_tol, w_threshold.
                max_iter defines the maximum training iterations; h_tol specifies the dagness tolerance; and w_threshold
                specifies the threshold for zeroing the entries of the final weighted adjacency matrix at post-processing.
        """
        assert isinstance(dataset, TemporalDataset)
        assert dataset.train_segmentation is not None
        assert train_config_dict is not None

        # Get the training configs
        max_iter = train_config_dict.get("max_iter", 1000)
        h_tol = train_config_dict.get("h_tol", 1e-8)
        w_threshold = train_config_dict.get("w_threshold", 0.4)
        # Get the numpy data
        data, _ = dataset.train_data_and_mask
        train_seg = dataset.train_segmentation
        # Loop over segmentation to create a list of pandas dataframes. Each contains a single time series
        dataframe_list = []
        for seg in train_seg:
            start_idx, end_idx = seg
            dataframe_list.append(pd.DataFrame(data[start_idx : end_idx + 1, :]))  # [series_len, num_variables]

        # Fit the model
        self.learner = from_pandas_dynamic(
            dataframe_list,
            p=self.lag,
            lambda_w=self.lambda_w,
            lambda_a=self.lambda_a,
            max_iter=max_iter,
            h_tol=h_tol,
            w_threshold=w_threshold,
        )
        assert isinstance(self.learner, StructureModel)

    def get_adj_matrix(self, do_round: bool = True, samples: int = 100, most_likely_graph: bool = False) -> np.ndarray:
        """
        This will return the learned adj matrix [lag+1, from, to]. Since the original learner does not support the temporal adj matrix.
        We need to leverage networkx for adj matrix conversion and post process it to the correct format. The adj format after nextworkx
        is [(lag+1)*num_nodes, (lag+1)*num_nodes], where adj[0:lag+1,...] (assume lag=2) specifies node1_lag0, node1_lag1, node1_lag2.
        Returns:
            np.ndarray: The learned temporal adj matrix [lag+1, num_nodes, num_nodes]
        """
        _ = do_round
        _ = samples
        _ = most_likely_graph  # Not used, just to make mypy happy.
        adj_static = nx.to_numpy_array(self.learner)  # [(lag+1)*num_nodes, (lag+1)*num_nodes]

        temporal_adj_list = []
        for lag in range(self.lag + 1):
            cur_adj = adj_static[lag :: self.lag + 1, 0 :: self.lag + 1]
            temporal_adj_list.append(cur_adj)

        temporal_adj = np.stack(temporal_adj_list, axis=0)  # [lag+1, num_nodes, num_nodes]
        temporal_adj = (temporal_adj != 0).astype(int)

        return temporal_adj

    def save(self) -> None:
        # Save variables
        os.makedirs(self.save_dir, exist_ok=True)
        self.variables.save(os.path.join(self.save_dir, self._variables_path))
        # Save model in pickle format
        # It is easier to just store the entire model as .pkl
        model_path = os.path.join(self.save_dir, self.model_file)
        logger.info(f"saved model to {model_path}")
        with open(f"{model_path}", "wb") as f:
            pickle.dump(self.learner, f)

    @classmethod
    def load(cls, model_id: str, save_dir: str, device: Union[str, int]) -> "Dynotears":
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
