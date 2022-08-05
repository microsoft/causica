# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch

from ...datasets.dataset import CausalDataset, Dataset
from ...datasets.variables import Variables
from ...models.deci.deci import DECI
from .end2end_causal import End2endCausal

T = TypeVar("T", bound="InformedDECI")


class InformedDECI(End2endCausal):
    """
    Wrapper class the true graph as a soft prior for DECI discovery + DECI inference
    The prior is enforced with a penalty:  self.lambda_prior * (self.prior_mask * (A - self.prior_A_confidence * self.prior_A)).abs().sum()
    where A is the learnt DAG, prior_A is a potentially non-dag adjacency matrix, prior_A_confidence (in [0,1]) represents our beliefs about the
    presence of edges and lambda_prior (in R+) represents the strength of our prior beliefs.
    """

    def __init__(self, model_id: str, variables: Variables, save_dir: str, device: torch.device, **model_config_dict):
        """
        model_id: Unique model ID for referencing this model instance.
        variables: Information about variables/features used by this model.
        save_dir: Location to save any information about this model, including training data. This wrapper is stateless but this information is needed for test result saving purposes.
        device: device on which DECI will be run.
        model_config_dict: nested dictionary containing extra arguments for DECI
        """

        self.discovery_config, self.inference_config = self.split_configs(model_config_dict)
        discovery_model = None
        inference_model = DECI(model_id, variables, save_dir, device, **self.inference_config)
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
        assert isinstance(dataset, CausalDataset)
        true_graph = dataset.get_adjacency_data_matrix().astype(np.float32)
        true_mask = dataset.get_known_subgraph_mask_matrix().astype(np.float32)
        assert isinstance(self.inference_model, DECI)
        self.inference_model.prior_A.data = torch.tensor(true_graph).to(
            dtype=self.inference_model.prior_A.dtype, device=self.inference_model.prior_A.device
        )
        self.inference_model.prior_mask.data = torch.tensor(true_mask).to(
            dtype=self.inference_model.prior_mask.dtype, device=self.inference_model.prior_mask.device
        )
        self.inference_model.exist_prior = True

        self.inference_model.run_train(
            dataset=dataset,
            train_config_dict=inference_config,
            report_progress_callback=report_progress_callback,
        )

    def get_adj_matrix(self, do_round: bool = True, samples: int = 100, most_likely_graph: bool = False):
        """
        Returns adjacency learnt by DECI (the discovery-inference model) as a numpy array
        """
        assert isinstance(self.inference_model, DECI)
        return self.inference_model.get_adj_matrix(
            do_round=do_round, samples=samples, most_likely_graph=most_likely_graph
        )

    # TODO: remove this method, use parent's implementation and change parent's implementation to create object of class it was called on by using relfection
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
        return super().create(
            model_id=model_id,
            save_dir=save_dir,
            variables=variables,
            model_config_dict=model_config_dict,
            device=device,
        )

    @classmethod
    def name(cls) -> str:
        return "informed_deci"
