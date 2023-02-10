from typing import List, Optional, Tuple

import torch
import torch.distributions as td
from torch.nn import Parameter, ParameterList

from causica.distributions.noise_accessible.noise_accessible import NoiseAccessible
from causica.distributions.splines.rational_quadratic_transform import PiecewiseRationalQuadraticTransform

SplineParamListType = List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]


class SplineDistribution(td.TransformedDistribution, NoiseAccessible):
    """A Spline Based Noise Distribution."""

    def __init__(
        self,
        base_loc: torch.Tensor,
        base_scale: torch.Tensor,
        param_list: SplineParamListType,
        output_bias: Optional[torch.Tensor] = None,
    ):
        self.base_loc = base_loc
        self.base_scale = base_scale
        self.param_list = param_list

        output_affine_transform: List[td.Transform] = []
        if output_bias is not None:
            output_affine_transform = [
                td.AffineTransform(loc=output_bias, scale=torch.ones_like(output_bias, device=output_bias.device))
            ]

        super().__init__(
            base_distribution=td.Normal(loc=self.base_loc, scale=self.base_scale),
            transforms=[_create_composite_layer(*layer) for layer in param_list] + output_affine_transform,
        )

    def sample_to_noise(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Transform from the sample observations to corresponding noise variables.

        Args:
            samples: Tensor of shape sample_shape + batch_shape + event_shape
        Returns:
            The generated samples with shape sample_shape + batch_shape + event_shape
        """
        output_affine_transform = self.transforms[-1]
        return output_affine_transform.inv(samples)

    def noise_to_sample(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Generate samples using the given exogenous noise.

        Args:
            noise: noise variable with shape sample_shape + batch_shape.
        Returns:
            The generated samples with shape sample_shape + batch_shape + event_shape
        """
        output_affine_transform = self.transforms[-1]
        return output_affine_transform(noise)


def create_spline_dist_params(
    features: int, flow_steps: int = 1, init_scale: float = 1e-2, num_bins: int = 8
) -> ParameterList:
    param_list = ParameterList()
    for i in range(flow_steps + 1):
        scale = Parameter(
            torch.zeros(features), requires_grad=True
        )  # this will be exponentiated when passed to the spline distribution
        loc = Parameter(torch.zeros(features), requires_grad=True)
        if i == flow_steps:
            knot_locations = None
            derivatives = None
        else:
            knot_locations = Parameter(init_scale * torch.randn(features, num_bins, 2))
            derivatives = Parameter(init_scale * torch.randn(features, num_bins - 1))
        param_list.append(ParameterList([loc, scale, knot_locations, derivatives]))
    return param_list


def _create_composite_layer(
    loc: torch.Tensor,
    scale: torch.Tensor,
    knot_locations: Optional[torch.Tensor],
    derivatives: Optional[torch.Tensor],
) -> td.Transform:
    """Create the composite layers as required by the spline distribution."""
    affine = td.AffineTransform(loc=loc, scale=scale.exp())
    if knot_locations is None:
        return affine
    return td.ComposeTransform([affine, PiecewiseRationalQuadraticTransform(knot_locations, derivatives)])
