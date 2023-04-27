from typing import Optional, Union

import torch
import torch.distributions as td
from torch import nn

from causica.distributions.noise.noise import IndependentNoise, Noise, NoiseModule
from causica.distributions.noise.spline.rational_quadratic_transform import PiecewiseRationalQuadraticTransform

# Ordered inputs to `_create_composite_layer`
SplineParams = tuple[torch.Tensor, ...]


class SplineNoise(td.TransformedDistribution, Noise):
    """A Spline Based Noise Distribution.

    Parametrized as in in [Neural Spline Flows](https://arxiv.org/pdf/1906.04032.pdf).
    """

    def __init__(
        self,
        base_loc: torch.Tensor,
        base_scale: torch.Tensor,
        spline_transforms: list[Union[td.AffineTransform, td.ComposeTransform]],
    ):
        """
        Args:
            base_loc: Loc of base normal distribution.
            base_scale: Scale of base normal distribution.
            spline_transforms: Spline transforms, where the last transform bijects from noise to samples.
        """
        self.base_loc = base_loc
        self.base_scale = base_scale
        last_transform = spline_transforms[-1]
        if spline_transforms and not (isinstance(last_transform, td.AffineTransform) or not last_transform.parts):
            # `td.identity` is just a `td.ComposeTransform` with no `parts`. It's used when there's no output shift.
            raise TypeError(
                "The last transformation must be `td.AffineTransform` or td.identity, but was "
                f"`{type(last_transform)}`."
            )
        super().__init__(
            base_distribution=td.Normal(loc=self.base_loc, scale=self.base_scale),
            transforms=spline_transforms,
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
    dim: int,
    num_bins: int,
    flow_steps: int,
    knot_locations_scale: float,
    derivatives_scale: float,
) -> list[SplineParams]:
    """Create initial values for a spline distribution.

    Args:
        dim: Number of dimensions of the represented variable.
        num_bins: Number of spline bins.
        flow_steps: Number of flow steps.
        knot_locations_scale: Scale of random values used for `knot_locations` of `PiecewiseRationalQuadraticTransform`.
        derivatives_scale: Scale of random values for `derivatives` of `PiecewiseRationalQuadraticTransform`.

    Returns:
        A list of parameters for `CompositeSplineLayer`s.
    """
    param_list: list[SplineParams] = []
    for i in range(flow_steps + 1):
        log_scale = torch.zeros(dim)  # this will be exponentiated when passed to the spline distribution
        loc = torch.zeros(dim)
        if i == flow_steps:
            param_list.append((loc, log_scale))
        else:
            knot_locations = knot_locations_scale * torch.randn(dim, num_bins, 2)
            derivatives = derivatives_scale * torch.randn(dim, num_bins - 1)
            param_list.append((loc, log_scale, knot_locations, derivatives))
    return param_list


class CompositeSplineLayer(nn.Module):
    """A layer constructing an affine transformation potentially composed with a `PiecewiseRationalQuadraticTransform`.

    Encapsulates one of the layers or `flow_steps` of the [Neural Spline Flows](https://arxiv.org/pdf/1906.04032.pdf).
    """

    def __init__(
        self,
        init_loc: torch.Tensor,
        init_log_scale: torch.Tensor,
        init_knot_locations: Optional[torch.Tensor] = None,
        init_derivatives: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            init_loc: Initial value of base normal distribution loc.
            init_scale: Initial value of base normal distribution scale.
            init_knot_locations: Initialization value for the knot locations of `PiecewiseRationalQuadraticTransform`.
            init_derivatives: Initialization value for the derivatives of `PiecewiseRationalQuadraticTransform`.
        """
        super().__init__()
        self.loc = nn.Parameter(init_loc)
        self.log_scale = nn.Parameter(init_log_scale)
        if (init_knot_locations is None) != (init_derivatives is None):
            raise ValueError("Either both or none of `knot_location` and `derivatives` must be set.")
        self.knot_locations = nn.Parameter(init_knot_locations) if init_knot_locations is not None else None
        self.derivatives = nn.Parameter(init_derivatives) if init_derivatives is not None else None

    def __call__(self, *args, **kwargs) -> td.Transform:
        return super().__call__(*args, **kwargs)

    def forward(self) -> td.Transform:
        affine = td.AffineTransform(loc=self.loc, scale=torch.exp(self.log_scale))
        if self.knot_locations is None or self.derivatives is None:  # Test both to ensure correct typing below
            return affine
        return td.ComposeTransform([affine, PiecewiseRationalQuadraticTransform(self.knot_locations, self.derivatives)])


class SplineNoiseModule(NoiseModule[IndependentNoise[SplineNoise]]):
    """Implements Neural Spline Flow noise with learnable parameters.

    See [Neural Spline Flows](https://arxiv.org/pdf/1906.04032.pdf).
    """

    def __init__(
        self,
        dim: int,
        num_bins: int = 8,
        flow_steps: int = 1,
        init_knot_locations_scale: float = 1e-2,
        init_derivatives_scale: float = 1e-2,
    ):
        """
        Args:
            dim: Number of dimensions of the represented variable.
            num_bins: Number of spline bins.
            flow_steps: Number of flow steps.
            init_knot_locations_scale: Scale of random initialization values for `knot_locations` of
                                       `PiecewiseRationalQuadraticTransform`.
            init_derivatives_scale: Scale of random initialization values for `derivatives` of
                                    `PiecewiseRationalQuadraticTransform`.
        """
        super().__init__()
        spline_params = create_spline_dist_params(
            dim=dim,
            num_bins=num_bins,
            flow_steps=flow_steps,
            knot_locations_scale=init_knot_locations_scale,
            derivatives_scale=init_derivatives_scale,
        )
        self.composite_spline_layers = nn.ModuleList(CompositeSplineLayer(*params) for params in spline_params)
        self.base_scale: torch.Tensor
        self.base_loc: torch.Tensor
        self.register_buffer("base_loc", torch.zeros(dim))
        self.register_buffer("base_scale", torch.ones(dim))

    def forward(self, x: Optional[torch.Tensor] = None) -> IndependentNoise[SplineNoise]:
        transforms = [spline_layer() for spline_layer in self.composite_spline_layers]
        if x is not None:
            transforms.append(td.AffineTransform(loc=x, scale=torch.ones_like(x, device=x.device)))
        else:
            # Maintain the old behavior of having this transform biject between noise and samples.
            transforms.append(td.identity_transform)

        spline_dist = SplineNoise(base_loc=self.base_loc, base_scale=self.base_scale, spline_transforms=transforms)
        return IndependentNoise(spline_dist, 1)
