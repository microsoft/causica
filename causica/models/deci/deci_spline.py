from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ...datasets.variables import Variables
from .deci import DECI


class DECISpline(DECI):
    """
    Flow-based VISL model where additive noise SEM base distribution is fixed to being a learnable spline distribution. Does causal discovery using a contractive and
    invertible GNN. The adjacency is a random variable over which we do inference.
    """

    def __init__(
        self,
        model_id: str,
        variables: Variables,
        save_dir: str,
        device: torch.device,
        imputation: bool = True,
        lambda_dag: float = 1.0,
        lambda_sparse: float = 1.0,
        tau_gumbel: float = 1.0,
        spline_bins: int = 16,
        var_dist_A_mode: str = "enco",
        imputer_layer_sizes: Optional[List[int]] = None,
        mode_adjacency: str = "learn",
        norm_layers: bool = False,
        res_connection: bool = False,
        encoder_layer_sizes: Optional[List[int]] = None,
        decoder_layer_sizes: Optional[List[int]] = None,
        cate_rff_n_features: int = 3000,
        cate_rff_lengthscale: Union[int, float, List[float], Tuple[float, float]] = (0.1, 1),
        prior_A: Union[torch.Tensor, np.ndarray] = None,
        prior_A_confidence: float = 0.5,
    ):
        """
        Args:
            model_id: Unique model ID for referencing this model instance.
            variables: Information about variables/features used by this model.
            save_dir: Location to save any information about this model, including training data.
            device: Device to load model to.
            imputation: Whether to train an imputation network simultaneously with the DECI network.
            lambda_dag: Coefficient for the prior term that enforces DAG.
            lambda_sparse: Coefficient for the prior term that enforces sparsity.
            tau_gumbel: Temperature for the gumbel softmax trick.
            spline_bins: How many bins to use for spline flow base distribution if the 'spline' choice is made
            var_dist_A_mode: Variational distribution for adjacency matrix. Admits {"simple", "enco", "true"}. "simple"
                             parameterizes each edge (including orientation) separately. "enco" parameterizes
                             existence of an edge and orientation separately. "true" uses the true graph.
            imputer_layer_sizes: Number and size of hidden layers for imputer NN for variational distribution.
            mode_adjacency: In {"upper", "lower", "learn"}. If "learn", do our method as usual. If
                            "upper"/"lower" fix adjacency matrix to strictly upper/lower triangular.
            norm_layers: bool indicating whether all MLPs should use layer norm
            res_connection:  bool indicating whether all MLPs should use layer norm
            encoder_layer_sizes: Optional list indicating width of layers in GNN encoder MLP
            decoder_layer_sizes: Optional list indicating width of layers in GNN decoder MLP
            cate_rff_n_features: number of random features to use in functiona pproximation when estimating CATE,
            cate_rff_lengthscale: lengthscale of RBF kernel used when estimating CATE,
            prior_A: prior adjacency matrix,
            prior_A_confidence: degree of confidence in prior adjacency matrix enabled edges between 0 and 1,
        """

        super().__init__(
            model_id=model_id,
            variables=variables,
            save_dir=save_dir,
            device=device,
            imputation=imputation,
            lambda_dag=lambda_dag,
            lambda_sparse=lambda_sparse,
            tau_gumbel=tau_gumbel,
            base_distribution_type="spline",
            spline_bins=spline_bins,
            var_dist_A_mode=var_dist_A_mode,
            imputer_layer_sizes=imputer_layer_sizes,
            mode_adjacency=mode_adjacency,
            norm_layers=norm_layers,
            res_connection=res_connection,
            encoder_layer_sizes=encoder_layer_sizes,
            decoder_layer_sizes=decoder_layer_sizes,
            cate_rff_n_features=cate_rff_n_features,
            cate_rff_lengthscale=cate_rff_lengthscale,
            prior_A=prior_A,
            prior_A_confidence=prior_A_confidence,
        )

    @classmethod
    def name(cls) -> str:
        return "deci_spline"
