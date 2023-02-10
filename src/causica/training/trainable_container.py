from enum import Enum
from typing import Dict, Optional, Tuple

import torch
from torch.nn import Parameter, ParameterDict

from causica.distributions import (
    NoiseAccessibleBernoulli,
    NoiseAccessibleCategorical,
    NoiseAccessibleIndependent,
    NoiseAccessibleMultivariateNormal,
    ParametrizedDistribution,
    SplineDistribution,
    create_spline_dist_params,
)
from causica.functional_relationships import ICGNN


class NoiseDist(Enum):
    SPLINE = "spline"
    GAUSSIAN = "gaussian"


class TrainableContainer(torch.nn.Module):
    """
    A module to contain all of the submodules associated with training on the CSuite example.

    This allows the state of the computation to be easily saved and loaded.
    """

    def __init__(
        self,
        dataset_name: str,
        icgnn: ICGNN,
        vardist: ParametrizedDistribution,
        noise_dist_params: ParameterDict,
        noise_dist_type: NoiseDist,
    ):
        """
        Args:
            dataset_name: the name of the CSuite dataset to train on
            icgnn: the ICGNN module
            vardist: The variational distribution that approximates the posterior
            noise_dist: The module of the parameters of the noise distribution for the SEM
            noise_dist_type: The continuous noise distribution used
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.icgnn = icgnn
        self.vardist = vardist
        self.noise_dist = noise_dist_params
        self.noise_dist_type = noise_dist_type


def create_noise_dists(
    shapes: Dict[str, torch.Size],
    types_dict: Dict[str, str],
    noise_dist: NoiseDist,
    noise_dist_params: Optional[ParameterDict] = None,
) -> Tuple[Dict, ParameterDict]:
    """Create the noise distribution for each node along with its associated parameters."""
    param_dict = ParameterDict({}) if noise_dist_params is None else noise_dist_params
    noise_dist_funcs = {}
    for key, shape_tuple in shapes.items():
        shape = shape_tuple[-1]
        var_type = types_dict[key]
        if var_type == "categorical":
            if key not in param_dict:
                param_dict[key] = ParameterDict({"base_logits": Parameter(torch.zeros(shape), requires_grad=True)})
            noise_dist_funcs[key] = lambda x, params=param_dict[key]: NoiseAccessibleCategorical(
                delta_logits=x, **params
            )
        elif var_type == "binary":
            if key not in param_dict:
                param_dict[key] = ParameterDict({"base_logits": Parameter(torch.zeros(shape), requires_grad=True)})
            noise_dist_funcs[key] = lambda x, params=param_dict[key]: NoiseAccessibleIndependent(
                NoiseAccessibleBernoulli(delta_logits=x, **params), 1
            )
        # continuous variables
        elif noise_dist == NoiseDist.SPLINE:
            if key not in param_dict:
                noise_param_list = create_spline_dist_params(features=shape, flow_steps=1)
                base_scale = Parameter(torch.ones(shape), requires_grad=False)
                base_loc = Parameter(torch.zeros(shape), requires_grad=False)
                param_dict[key] = ParameterDict(
                    dict(base_loc=base_loc, base_scale=base_scale, param_list=noise_param_list)
                )
            noise_dist_funcs[key] = lambda x, params=param_dict[key]: NoiseAccessibleIndependent(
                SplineDistribution(output_bias=x, **params), 1
            )
        elif noise_dist == NoiseDist.GAUSSIAN:
            if key not in param_dict:
                param_dict[key] = ParameterDict({"log_scale": torch.zeros([shape], requires_grad=True)})
            noise_dist_funcs[key] = lambda x, params=param_dict[key]: NoiseAccessibleMultivariateNormal(
                loc=x, scale_tril=torch.diag(torch.exp(params["log_scale"]))
            )
        else:
            raise ValueError("Unrecognised model", noise_dist)

    return noise_dist_funcs, param_dict
