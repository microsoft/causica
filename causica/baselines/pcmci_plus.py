from __future__ import annotations

import logging
import os
import pickle
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests import GPDC, CMIknn, CMIsymb, ParCorr
from tigramite.pcmci import PCMCI

from ..datasets.dataset import Dataset, TemporalDataset
from ..datasets.variables import Variables
from ..models.imodel import IModelForCausalInference
from ..models.model import Model
from ..models.torch_model import _set_random_seed_and_remove_from_config
from ..utils.causality_utils import cpdag2dags
from ..utils.io_utils import read_json_as, save_json, save_txt
from ..utils.nri_utils import convert_temporal_to_static_adjacency_matrix

T = TypeVar("T", bound="PCMCI_Plus")
logger = logging.getLogger(__name__)


class PCMCI_Plus(Model, IModelForCausalInference):
    _model_config_path = "model_config.json"
    _model_type_path = "model_type.txt"
    _variables_path = "variables.json"
    model_file = "model.pkl"
    result_file = "results.json"

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        max_lag: int,
        cond_ind_test: str,
        verbosity: int = 0,
        cond_ind_test_config: Optional[dict] = None,
        mec_mode: str = "enumerate",
    ):
        """
        This initialises a pcmci_plus instance.
        Args:
            model_id: str, the id of the model
            variables: Variables, the variables corresponds to the training data.
            save_dir: str, the directory to save the model and relevant outputs.
            max_lag: the maximum lag it considers.
            cond_ind_test: a str specifies which conditional independence test to run. It supports (1) "parcorr", (2) "gpdc", (3) "cmiknn", and (4) "cmisymb".
                (1) "parcorr" is the partial correlation test using linear ordinary least squares regression.; (2) "gpdc" is the Gaussian Process regression
                based conditional independence test; (3) "cmiknn" is the conditional mutual information test using k-nearest neighbor estimator; (4) "cmisymb" is the
                conditional mutual information test based on discrete estimator.
            verbosity: the level of information to display for debugging.
            cond_ind_test_config: a dict of configs for the conditional independence test. If none, default settings are used.
                For the details, please refer to https://jakobrunge.github.io/tigramite/ .
            mec_mode: "truth": for bidirected edges, we assume it is the same as truth graph. "enumerate": it enumerates all possible DAGs within
                the MEC.
        """
        super().__init__(model_id, variables, save_dir)
        self.max_lag = max_lag
        assert cond_ind_test in ("parcorr", "gpdc", "cmiknn", "cmisymb"), "Unsupported conditional independence test."
        assert mec_mode in ("enumerate", "truth")
        self.mec_mode = mec_mode

        # Get conditional independence test configs.
        self.cond_ind_test_config = cond_ind_test_config
        cond_test_dict = {"parcorr": ParCorr, "gpdc": GPDC, "cmiknn": CMIknn, "cmisymb": CMIsymb}

        if self.cond_ind_test_config is not None:
            self.cond_ind_test = cond_test_dict[cond_ind_test](**self.cond_ind_test_config)
        else:
            self.cond_ind_test = cond_test_dict[cond_ind_test]()

        self.verbosity = verbosity
        self.learner: Any = None
        # this is to save the results from the learner
        self.results: Optional[dict] = None
        self.dataset: Optional[TemporalDataset] = None

    @classmethod
    def name(cls) -> str:
        return "pcmci_plus"

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

        # Remove the random_seed entry
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

        assert isinstance(dataset, TemporalDataset), "PCMCI_Plus only supports temporal datasets."
        assert dataset.train_segmentation is not None
        assert train_config_dict is not None

        # this is stored for adj_matrix processing
        self.dataset = dataset

        # Get the training configs
        pc_alpha = train_config_dict.get("pc_alpha", 0.01)

        # Get the numpy data
        data, _ = dataset.train_data_and_mask
        train_seg = dataset.train_segmentation

        # convert each time series to a tigramite dataframe
        tigramite_data_dict = {}
        for series_idx, seg in enumerate(train_seg):
            start_idx, end_idx = seg
            tigramite_data_dict[f"series_{series_idx}"] = data[start_idx : end_idx + 1, :]

        if len(tigramite_data_dict) > 1:
            time_offset = {key: 0 for key, _ in tigramite_data_dict.items()}
            tigramite_dataframe = pp.DataFrame(tigramite_data_dict, analysis_mode="multiple", time_offsets=time_offset)
        else:
            tigramite_dataframe = pp.DataFrame(tigramite_data_dict, analysis_mode="single")

        # run pcmci_plus
        self.learner = PCMCI(tigramite_dataframe, cond_ind_test=self.cond_ind_test, verbosity=self.verbosity)
        self.results = self.learner.run_pcmciplus(tau_min=0, tau_max=self.max_lag, pc_alpha=pc_alpha)

    def get_adj_matrix(self, do_round: bool = True, samples: int = 100, most_likely_graph: bool = False) -> np.ndarray:
        """
        This will extract the temporal adjacency matrix from self.results. However, this can only give MEC for inst effect.
        There are two modes for resolving it: (1) "truth": for all ambiguities, it assigns to ground truth. (2) "enumerate": it considers all possible graphs within the MEC.
        However, for all conflict edges, we discard them.

        Returns:
            temporal_adj_matrix: np.ndarray with shape [lag+1, num_nodes, num_nodes] or [num_possible_DAG, lag+1, num_nodes, num_nodes]
        """
        # pylint: disable=unused-variable
        _ = do_round
        _ = samples
        _ = most_likely_graph  # Not used, just to make mypy happy.

        assert self.results is not None
        # process the raw adj output
        adj_matrix = self._process_adj_matrix(
            self.results["graph"]
        )  # shape [lag+1, num_nodes, num_nodes], and adj_matrix[0] is a cpdag
        adj_matrix = self._process_cpdag(
            adj_matrix
        )  # either [lag+1, num_nodes, num_nodes] (mec_mode = "truth") or [all_posible_dag, lag+1, num_nodes, num_nodes] (mec_mode = "enumerate")

        return adj_matrix

    def _process_cpdag(self, adj_matrix: np.ndarray):
        """
        This will process the inst cpdag (i.e. adj_matrix[0, ...]) according to the mec_mode. It supports "enumerate" and "truth"
        Args:
            adj_matrix: np.ndarray, a temporal adj matrix with shape [lag+1, num_nodes, num_nodes] where the inst part can be a cpdag.

        Returns:
            adj_matrix: np.ndarray with shape [num_possible_dags, lag+1, num_nodes, num_nodes] (mec_mode = enumerate) or [lag+1, num_nodes, num_nodes] (mec_mode=truth)
        """
        if self.mec_mode == "enumerate":
            lag_plus, num_nodes = adj_matrix.shape[0], adj_matrix.shape[1]
            static_temporal_graph = convert_temporal_to_static_adjacency_matrix(
                adj_matrix, conversion_type="auto_regressive"
            )  # shape[(lag+1) *nodes, (lag+1)*nodes]
            all_static_temp_dags = cpdag2dags(
                static_temporal_graph, samples=3000
            )  # [all_possible_dags, (lag+1)*num_nodes, (lag+1)*num_nodes]
            # convert back to temporal adj matrix.
            temp_adj_list = np.split(
                all_static_temp_dags[..., :, (lag_plus - 1) * num_nodes :], lag_plus, axis=1
            )  # list with length lag+1, each with shape [all_possible_dags, num_nodes, num_nodes]
            proc_adj_matrix = np.stack(
                list(reversed(temp_adj_list)), axis=-3
            )  # shape [all_possible_dags, lag+1, num_nodes, num_nodes]
        elif self.mec_mode == "truth":
            # we must have dataset and corresponding true adj to assign to bi-directed edges.
            assert isinstance(self.dataset, TemporalDataset)
            assert self.dataset.has_adjacency_data_matrix, "dataset does not have ground truth adj matrix"
            assert (
                self.dataset.get_adjacency_data_matrix().ndim == 2
            ), "truth mec_mode does not support aggregated adj matrix"
            proc_adj_matrix = deepcopy(adj_matrix)
            ground_truth = self.dataset.get_adjacency_data_matrix()
            proc_adj_matrix[0, ...] = self._assign_true_edges(
                proc_adj_matrix[0, ...], ground_truth[0, ...]
            )  # shape [lag+1, num_nodes, num_nodes]
        else:
            raise ValueError

        return proc_adj_matrix

    def _assign_true_edges(self, inst_pred: np.ndarray, inst_true: np.ndarray) -> np.ndarray:
        """
        This will compare the bi-directed edges in inst_pred and compare with inst_true, if inst_true has a directed edge, we assign
        this true direction to the inst_pred, if inst_true does not have an edge, we choose the lower part of the bi-directed edge and assign the direction.
        E.g. i>j, inst_pred[i,j] = 1 and inst_pred[j,i] = 1 but inst_true[i,j] and inst_true[j,i]=0, we assign inst_pred[i,j] = 1 and inst_pred[j,i] = 0.
        If inst_true[i,j]=1 and inst_true[j,i]=0, we assign inst_pred[i,j] = 1 and inst_pred[j,i] = 0.
        Args:
            inst_pred: The cpdag of inst effect from the model with shape [num_nodes, num_nodes]
            inst_true: The ground truth inst effect adj matrix with shape [num_nodes, num_nodes]

        Returns:
            proc_inst_pred: the processed inst adj matrix with shape [num_nodes, num_nodes]
        """
        assert inst_pred.ndim == 2 and inst_true.ndim == 2

        bidirectional_mask = inst_pred * inst_pred.T
        converted_adj = (1 - bidirectional_mask) * inst_pred
        converted_adj += bidirectional_mask * inst_true
        converted_adj += bidirectional_mask * np.tril((1 - inst_true) * (1 - inst_true).T)

        return converted_adj

    def _process_adj_matrix(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        This will process the raw output adj graphs from pcmci_plus. The raw output can contain 3 types of edges:
            (1) "-->" or "<--". This indicates the directed edges, and they should appear symmetrically in the matrix.
            (2) "o-o": This indicates the bi-directed edges, also appears symmetrically.
            Note: for lagged matrix, it can only contain "-->".
            (3) "x-x": this means the edge direction is un-decided due to conflicting orientation rules. We ignores
                the edges in this case.
        Args:
            inst_matrix: the input raw inst matrix with shape [num_nodes, num_nodes, lag+1]

        Returns:
            inst_adj_matrix: np.ndarray, an inst adj matrix with shape [lag+1, num_nodes, num_nodes]
        """
        assert adj_matrix.ndim == 3

        adj_matrix = deepcopy(adj_matrix)
        adj_matrix = np.moveaxis(adj_matrix, -1, 0)  # shape [lag+1, num_nodes, num_nodes]
        adj_matrix[adj_matrix == ""] = 0
        adj_matrix[adj_matrix == "<--"] = 0
        adj_matrix[adj_matrix == "-->"] = 1
        adj_matrix[adj_matrix == "o-o"] = 1
        adj_matrix[adj_matrix == "x-x"] = 0

        return adj_matrix.astype(float)

    def save(self) -> None:
        # Save variables
        os.makedirs(self.save_dir, exist_ok=True)
        self.variables.save(os.path.join(self.save_dir, self._variables_path))
        # Save model in pickle format
        # It is easier to just store the entire model as .pkl
        model_path = os.path.join(self.save_dir, self.model_file)
        result_path = os.path.join(self.save_dir, self.result_file)
        logger.info(f"saved model to {model_path}")
        with open(f"{model_path}", "wb") as f:
            pickle.dump(self.learner, f)

        # Store the results if exists
        if self.results is not None:
            save_json(self.results, result_path)

    @classmethod
    def load(cls, model_id: str, save_dir: str, device: Union[str, int]) -> "PCMCI_Plus":
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

        # Load the results if exists
        result_path = os.path.join(save_dir, cls.result_file)
        if os.path.exists(result_path):
            model.results = read_json_as(result_path, dict)

        return model
