from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from ...datasets.dataset import CausalDataset, Dataset, LatentConfoundedCausalDataset
from ...datasets.variables import Variables
from ...models.deci.ddeci import ADMGParameterisedDDECI
from ...utils.causality_utils import admg2dag
from .end2end_causal import End2endCausal


class TrueGraphInformedADMGParameterisedDDECI(End2endCausal):
    def __init__(self, model_id: str, variables: Variables, save_dir: str, device=torch.device, **model_config_dict):
        self.discovery_config, self.inference_config = self.split_configs(model_config_dict)
        discovery_model = None
        inference_model = ADMGParameterisedDDECI(model_id, variables, save_dir, device, **self.inference_config)
        super().__init__(
            model_id=model_id,
            variables=variables,
            save_dir=save_dir,
            device=device,
            discovery_model=discovery_model,
            inference_model=inference_model,
            **model_config_dict,
        )

    def run_train(
        self,
        dataset: Dataset,
        train_config_dict: Optional[Dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:

        if train_config_dict is None:
            train_config_dict = {}
        _, inference_config = self.split_configs(train_config_dict)

        assert "prior_A_confidence" in self.inference_config.keys()
        assert isinstance(self.inference_model, ADMGParameterisedDDECI)

        if isinstance(dataset, LatentConfoundedCausalDataset):
            true_directed_adj = torch.as_tensor(dataset.get_directed_adjacency_data_matrix())
            true_bidirected_adj = torch.as_tensor(dataset.get_bidirected_adjacency_data_matrix())
        elif isinstance(dataset, CausalDataset):
            true_directed_adj = torch.as_tensor(dataset.get_adjacency_data_matrix())
            true_bidirected_adj = torch.zeros_like(true_directed_adj)
        else:
            raise TypeError

        true_adj = admg2dag(true_directed_adj, true_bidirected_adj).numpy().astype(np.float32)

        self.inference_model.prior_A.data = torch.tensor(true_adj).to(
            dtype=self.inference_model.prior_A.dtype, device=self.inference_model.prior_A.device
        )
        self.inference_model.exist_prior = True

        self.inference_model.run_train(
            dataset=dataset,
            train_config_dict=inference_config,
            report_progress_callback=report_progress_callback,
        )

    def get_adj_matrix(self, do_round: bool = True, samples: int = 100, most_likely_graph: bool = False):
        """Returns adjacency learnt by ADMGParameterisedDDECI as a numpy array."""
        assert isinstance(self.inference_model, ADMGParameterisedDDECI)
        return self.inference_model.get_adj_matrix(
            do_round=do_round, samples=samples, most_likely_graph=most_likely_graph
        )

    def get_admg_matrices(
        self, do_round: bool = True, samples: int = 100, most_likely_graph: bool = False, squeeze: bool = False
    ):
        """Returns the directed and bidirected adjaency matrices learnt by ADMGParameterisedDDECI as numpy arrays."""
        assert isinstance(self.inference_model, ADMGParameterisedDDECI)
        return self.inference_model.get_admg_matrices(
            do_round=do_round, samples=samples, most_likely_graph=most_likely_graph, squeeze=squeeze
        )

    @classmethod
    def name(cls) -> str:
        return "true_graph_informed_admg_ddeci"
