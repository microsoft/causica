from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ...datasets.dataset import Dataset, TemporalDataset
from ...datasets.temporal_tensor_dataset import TemporalTensorDataset
from ...datasets.variables import Variables
from ...utils.causality_utils import process_adjacency_mats
from ...utils.helper_functions import maintain_random_state, to_tensors
from ...utils.nri_utils import convert_temporal_to_static_adjacency_matrix
from ..torch_model import ONNXNotImplemented
from .deci import DECI

logger = logging.getLogger(__name__)


class FoldTimeDECI(DECI):
    _saved_sampled_adjacency_file = "saved_adjacency.npy"
    _saved_sampled_probable_adjacency_file = "saved_probable_adjacency.npy"

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        lag: int,
        allow_instantaneous: bool,
        treat_continuous: bool,
        imputation: bool = True,
        lambda_dag: float = 1.0,
        lambda_sparse: float = 1.0,
        lambda_prior: float = 1.0,
        tau_gumbel: float = 1.0,
        base_distribution_type: str = "spline",
        spline_bins: int = 8,
        var_dist_A_mode: str = "enco",
        mode_adjacency: str = "learn",
        norm_layers: bool = True,
        res_connection: bool = True,
        cate_rff_n_features: int = 3000,
        cate_rff_lengthscale: Union[float, List[float], Tuple[float, float]] = (0.1, 1.0),
        prior_A_confidence: float = 0.5,
        graph_constraint_matrix: Optional[np.ndarray] = None,
        dense_init: bool = False,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
        embedding_size: Optional[int] = None,
    ):
        """
        Init method to create fold-time DECI object.
        Args:
            model_id: Unique model ID for referencing this model instance.
            variables: Information about variables/features used by this model.
            save_dir: Location to save any information about this model, including training data.
            device: Device to load model to.
            lag: the user specified lag for fold-time DECI
            allow_instantaneous: whether fold-time DECI model the instantaneous effect
            treat_continuous: Whether we treat all variables as continuous variables
            Others: Model configs for DECI. Refer to DECI init for docs.
        """
        self.lag = lag
        assert lag > 0, "lag must be higher than 0"
        self.allow_instantaneous = allow_instantaneous
        self.treat_continuous = treat_continuous
        # This variable is the one inferred from temporal data.
        if self.treat_continuous:
            # Change the temporal variable type to continuous
            variables = self._change_variables_to_continuous(variables)

        self.variables_orig = variables
        # This is to create variables for fold-time format, s.t. DECI is created based on this fold-time variables.
        proc_variables = self._repeat_variables_for_fold_time(variables)
        super().__init__(
            model_id=model_id,
            variables=proc_variables,
            save_dir=save_dir,
            device=device,
            imputation=imputation,
            lambda_dag=lambda_dag,
            lambda_sparse=lambda_sparse,
            lambda_prior=lambda_prior,
            tau_gumbel=tau_gumbel,
            base_distribution_type=base_distribution_type,
            spline_bins=spline_bins,
            var_dist_A_mode=var_dist_A_mode,
            mode_adjacency=mode_adjacency,
            norm_layers=norm_layers,
            res_connection=res_connection,
            cate_rff_n_features=cate_rff_n_features,
            cate_rff_lengthscale=cate_rff_lengthscale,
            prior_A_confidence=prior_A_confidence,
            dense_init=dense_init,
            encoder_layer_sizes=encoder_layer_sizes,
            decoder_layer_sizes=decoder_layer_sizes,
            embedding_size=embedding_size,
        )
        # Build hard constraints
        if graph_constraint_matrix is None:
            # Build hard constraints
            hard_constraint = self._build_constraint_matrix()
        else:
            hard_constraint = graph_constraint_matrix
        self.hard_constraint = hard_constraint
        self.set_graph_constraint(self.hard_constraint)

    def set_prior_A(
        self, prior_A: Optional[Union[torch.Tensor, np.ndarray]], prior_mask: Optional[Union[torch.Tensor, np.ndarray]]
    ):
        """
        This enables the soft prior constraints by setting a prior and corresponding mask.
        Args:
            prior_A: Prior adjacency matrix for soft prior.
            prior_mask: The corresponding mask. Element 1 indicates soft prior are applied to this edge, and 0 allows
            the edge to be freely learned without soft prior constraint.
        """
        if prior_A is None:
            self.exist_prior = False
            self.prior_A = nn.Parameter(
                torch.zeros((self.num_nodes, self.num_nodes), device=self.device), requires_grad=False
            )
            self.prior_mask = nn.Parameter(
                torch.zeros((self.num_nodes, self.num_nodes), device=self.device), requires_grad=False
            )
        else:
            if isinstance(prior_A, np.ndarray):
                prior_lag = prior_A.shape[0] - 1
                assert prior_lag == self.lag, "Prior lag is not consistent with model lag."
                prior_A = convert_temporal_to_static_adjacency_matrix(prior_A, conversion_type="full_time").astype(
                    np.float_
                )
                prior_mask = convert_temporal_to_static_adjacency_matrix(
                    prior_mask, conversion_type="full_time", fill_value=1
                )
                prior_A = torch.Tensor(prior_A)
                assert prior_mask is not None
                prior_mask = torch.Tensor(prior_mask)

                self.prior_A.data = prior_A.to(dtype=self.prior_A.dtype, device=self.prior_A.device)
                assert prior_mask is not None
                self.prior_mask.data = prior_mask.to(dtype=self.prior_mask.dtype, device=self.prior_mask.device)
                self.exist_prior = True
            else:
                raise TypeError(f"{self.name()} only support np array prior matrix")

    def _repeat_variables_for_fold_time(self, variables: Variables) -> Variables:
        """
        This method will process the temporal variables before initializing the static DECI. It will change the variables from temporal format into fold-time format.
        E.g. Assume lag 1, temporal data with two variables ->  static data with 2 variables + 2 variables (copied from original variables).
        If treat_continuous is true, we process the type of variables as 'continuous'.
        Args:
            variables: Temporal variables.
        """
        assert len(variables.auxiliary_variables) == 0

        variable_list = []
        for cur_lag in range(self.lag + 1):
            for variable in variables:
                variable = deepcopy(variable)
                variable.name = f"Lag {self.lag-cur_lag} {variable.name}"
                variable.group_name = f"Lag {self.lag-cur_lag} {variable.group_name}"
                variable_list.append(variable)

        # Re-generate processed variables

        return Variables(variable_list, auxiliary_variables=None, used_cols=None)

    def _change_variables_to_continuous(self, variables: Variables):
        """
        This will change the type of each variable in variables to "continuous".
        Args:
            variables: variables
        """
        var_list = []
        for variable in variables:
            variable = deepcopy(variable)
            variable.type_ = "continuous"
            var_list.append(variable)
        proc_variables = Variables(var_list, auxiliary_variables=None, used_cols=None)

        return proc_variables

    def _create_dataset_for_deci(
        self, dataset: TemporalDataset, train_config_dict: Dict[str, Any]
    ) -> Tuple[DataLoader, int]:
        """
        Create a dataset class for fold-time data loading. This will return a customized dataloader which support the mini-batching for both fold-time
        and autoregressive models.
        """

        data, mask = self.process_dataset(dataset, train_config_dict)
        tensor_dataset = TemporalTensorDataset(
            *to_tensors(data, mask, device=self.device),
            lag=self.lag,
            is_autoregressive=False,
            index_segmentation=dataset.train_segmentation,
        )
        dataloader = DataLoader(tensor_dataset, batch_size=train_config_dict["batch_size"], shuffle=True)

        return dataloader, len(tensor_dataset)

    @classmethod
    def name(cls) -> str:
        return "fold_time_deci"

    def _build_constraint_matrix(self) -> np.ndarray:
        """
        This will build the hard constraint matrix so that the lagged effect cannot go against time.
        This constraint matrix can be divided into blocks with coarse shape [lag+1, lag+1]. E.g. if lag is 1 and number
        of node per time step is 3, then the constraint matrix will be [2 blocks, 2 blocks] (coarse structure), where each block has shape [3, 3].
        The coarse structure G_{ij} indicates the connections between time i and time j. Here, we adopt the representations:
        row 0 to row lag+1 represents t-lag to t, and the same goes for columns. So G_{0,1} in the above example indicates
        the connections between time t-1 and time t.

        For each block matrix, it specifies the node connections, where B_{ij} = 1 means the i -> j, where their time steps are specified by the block position
        in the constraint matrix.

        """

        num_nodes_tot = self.variables.num_groups
        lags = self.lag + 1
        num_node = num_nodes_tot // (lags)
        coarse_structure = np.triu(np.full((lags, lags), np.nan), k=0 if self.allow_instantaneous else 1)
        hard_constraint = np.kron(coarse_structure, np.eye(num_node))
        return np.float32(hard_constraint)

    def process_dataset(
        self,
        dataset: Dataset,
        train_config_dict: Optional[Dict[str, Any]] = None,
        variables: Optional[Variables] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Re-initialize the dataprocessor (which will be used in training) according to the original temporal variables
        if variables is None:
            variables = self.variables_orig
        return super().process_dataset(dataset, train_config_dict=train_config_dict, variables=variables)

    def save(self) -> None:
        """
        Save the torch model state_dict, as well as an ONNX representation of the model,
        if implemented.
        """
        self.variables_orig.save(os.path.join(self.save_dir, self._variables_path))
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

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Optional[Dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:

        # Set soft prior matrix for fold-time DECI
        assert isinstance(dataset, TemporalDataset)
        if train_config_dict is None:
            train_config_dict = {}
        # Run training
        super().run_train(dataset, train_config_dict, report_progress_callback)
        # Save the sampled adjacency matrix
        sampled_adjacency = self.get_adj_matrix(do_round=True, samples=100)
        proc_sampled_adjacency, _ = process_adjacency_mats(sampled_adjacency, num_nodes=sampled_adjacency.shape[-1])
        np.save(os.path.join(self.save_dir, self._saved_sampled_adjacency_file), proc_sampled_adjacency)

        sampled_probable_adjacency = self.get_adj_matrix(do_round=True, samples=1, most_likely_graph=True, squeeze=True)
        np.save(os.path.join(self.save_dir, self._saved_sampled_probable_adjacency_file), sampled_probable_adjacency)
