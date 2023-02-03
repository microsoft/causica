from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from ...datasets.dataset import Dataset
from ...datasets.variables import Variables
from ...models.deci.ddeci import ADMGParameterisedDDECI
from ..castle_causal_learner import CastleCausalLearner
from ..fci import FCI
from .end2end_causal import End2endCausal


class FCIInformedADMGParameterisedDDECI(End2endCausal):
    """Wrapper class for a random sampled graph returned by FCI as a prior for ADMGParameterisedDDECI."""

    def __init__(self, model_id: str, variables: Variables, save_dir: str, device: torch.device, **model_config_dict):
        """FCIInformedDDECI constructor.

        Args:
            model_id: Unique model ID for referencing this model instance.
            variables: Information about the variables/features used by this model.
            save_dir: Location to save any information about this model, including training data.
            device: Device on which ADMGParameterisedDDECI will be run.
        """
        self.discovery_config, self.inference_config = self.split_configs(model_config_dict)
        discovery_model = FCI(model_id, variables, save_dir, **self.discovery_config)
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
        train_config_dict: Optional[dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        if train_config_dict is None:
            train_config_dict = {}

        discovery_config, inference_config = self.split_configs(train_config_dict)
        assert self.discovery_model is not None
        self.discovery_model.run_train(
            dataset=dataset,
            train_config_dict=discovery_config,
            report_progress_callback=report_progress_callback,
        )

        assert isinstance(self.inference_model, ADMGParameterisedDDECI)
        assert isinstance(self.discovery_model, CastleCausalLearner)
        self.inference_model.prior_A.data = torch.tensor(self.discovery_model.get_adj_matrix(samples=1)).to(
            dtype=self.inference_model.prior_A.dtype, device=self.inference_model.device
        )
        self.inference_model.prior_mask.data = torch.ones_like(
            self.inference_model.prior_A.data,
            dtype=self.inference_model.prior_mask.dtype,
            device=self.inference_model.device,
        )
        self.inference_model.exist_prior = True
        assert "prior_A_confidence" in self.inference_config.keys()
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
        return "fci_informed_admg_ddeci"
