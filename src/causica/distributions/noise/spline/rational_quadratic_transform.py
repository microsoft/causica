"""
Implementation of the Piecewise Rational Quadratic Transform

This is pretty much a copy-paste of
    https://github.com/tonyduan/normalizing-flows/blob/master/nf/utils.py

We should consider using the Pyro implementation.
"""
from typing import Tuple

import numpy as np
import torch
import torch.distributions as td
from torch.nn import functional as F

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


class PiecewiseRationalQuadraticTransform(td.Transform):
    """
    Layer that implements a spline-cdf (https://arxiv.org/abs/1906.04032) transformation.

    All dimensions of x are treated as independent, no coupling is used. This is needed
    to ensure invertibility in our additive noise SEM.

    This is pretty much a copy-paste of
        https://github.com/tonyduan/normalizing-flows/blob/master/nf/utils.py
    """

    bijective = True

    def __init__(self, knot_locations: torch.Tensor, derivatives: torch.Tensor, tail_bound: float = 3.0):
        """
        Args:
            knot_locations: the x, y points of the knots,  shape [dim, num_bins, 2]
            derivatives: the derivatives at the knots, shape [dim, num_bins - 1]
            tail_bound: distance of edgemost bins relative to 0,
        """
        super().__init__()
        assert knot_locations.ndim == 3, "Only two dimensional params are supported"
        assert knot_locations.shape[-1] == 2, "Knot locations are 2-d"
        self.dim, self.num_bins, *_ = knot_locations.shape

        assert derivatives.shape == (self.dim, self.num_bins - 1)

        self.tail_bound = tail_bound
        self._event_dim = 0

        self.knot_locations = knot_locations
        self.derivatives = derivatives

    @property
    def event_dim(self):
        return self._event_dim

    def _piecewise_cdf(self, inputs: torch.Tensor, inverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the Cumulative Density function at `inputs`

        Args:
            inputs: the positions at which to evaluate the cdf shape batch_shape + (input_dim)
            inverse: whether this is forwards or backwards transform
        Returns:
            input_evaluations and absolute log determinants, a tuple of tensors of shape batch_shape + (input_dim)
        """
        assert len(inputs.shape) == 2  # TODO(JJ) accept 1d inputs
        batch_shape = inputs.shape[:-1]
        # shape batch_shape + (dim, 3 * num_bins - 1)
        expanded_knot_locations = self.knot_locations[None, ...].expand(*batch_shape, -1, -1, -1)

        unnormalized_derivatives = F.pad(
            self.derivatives[None, ...].expand(*batch_shape, -1, -1),
            pad=(1, 1),
            value=np.log(np.expm1(1 - DEFAULT_MIN_DERIVATIVE)),
        )  # shape batch_shape + (dim, num_bins + 1)

        inside_intvl_mask = torch.abs(inputs) <= self.tail_bound
        outputs = inputs.clone().detach()  # shape batch_shape + (input_dim)
        outputs[inside_intvl_mask] = 0

        logabsdet = torch.zeros_like(inputs)  # shape batch_shape + (input_dim)

        if inside_intvl_mask.any():
            outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = rational_quadratic_spline(
                inputs=inputs[inside_intvl_mask],
                unnormalized_knot_locations=expanded_knot_locations[inside_intvl_mask, :, :],
                unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
                inverse=inverse,
                tail_bound=self.tail_bound,
            )
        return outputs, logabsdet

    @td.constraints.dependent_property(is_discrete=False)
    def domain(self):
        if self.event_dim == 0:
            return td.constraints.real
        return td.constraints.independent(td.constraints.real, self.event_dim)

    @td.constraints.dependent_property(is_discrete=False)
    def codomain(self):
        if self.event_dim == 0:
            return td.constraints.real
        return td.constraints.independent(td.constraints.real, self.event_dim)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: batch_shape + (input_dim)
        Returns:
            transformed_input batch_shape + (input_dim)
        """
        return self._piecewise_cdf(x, inverse=False)[0]

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: batch_shape + (input_dim)
        Returns:
            transformed_input, batch_shape + (input_dim)
        """
        return self._piecewise_cdf(y, inverse=True)[0]

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: batch_shape + (input_dim)
        Returns:
            jacobian_log_determinant: batch_shape + (input_dim)
        """
        return self._piecewise_cdf(x, inverse=False)[1]


def rational_quadratic_spline(
    inputs: torch.Tensor,
    unnormalized_knot_locations: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    inverse: bool = False,
    tail_bound: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        inputs: shape batch_shape + (input_dim)
        unnormalized_knot_locations: shape batch_shape + (dim, num_bins, 2)
        unnormalized_derivatives: shape batch_shape + (dim, num_bins + 1)
        inverse: whether this is the forward or backward pass
        tail_bound:
    Returns:
        the outputs and the absolute log determinant, a tuple of tensors of shape batch_shape + (input_dim)
    """

    if (torch.abs(inputs) > tail_bound).any():
        raise ValueError("Input outside domain")

    num_bins = unnormalized_knot_locations.shape[-2]

    if DEFAULT_MIN_BIN_WIDTH * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if DEFAULT_MIN_BIN_HEIGHT * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    # create the bins
    derivatives = DEFAULT_MIN_DERIVATIVE + F.softplus(unnormalized_derivatives)
    # shape batch_shape + (dim)
    widths, cumwidths = normalize_bins(
        unnormalized_knot_locations[..., 0], tail_bound=tail_bound, min_bin_size=DEFAULT_MIN_BIN_WIDTH
    )
    heights, cumheights = normalize_bins(
        unnormalized_knot_locations[..., 1], tail_bound=tail_bound, min_bin_size=DEFAULT_MIN_BIN_HEIGHT
    )

    # place the input data within the bins
    bin_idx = torch.searchsorted(cumheights if inverse else cumwidths, inputs[..., None]) - 1

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    input_bin_heights = heights.gather(-1, bin_idx)[..., 0]

    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    # here be dragons
    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_bin_heights * (input_delta - input_derivatives)
        b = input_bin_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b * b - 4 * a * c
        assert (discriminant >= 0).all()

        theta = (2 * c) / (-b - torch.sqrt(discriminant))
        theta_one_minus_theta = theta * (1 - theta)

        outputs = theta * input_bin_widths + input_cumwidths

        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta * theta
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_bin_heights * (input_delta * theta * theta + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet


def normalize_bins(
    bins: torch.Tensor, tail_bound: float, min_bin_size: float = 1e-3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For a tensor of unnormalized bin widths `bins`, create bins in [-tail_bound, tail_bound]

    Args:
        bins: shape (..., num_bins)
        tail_bound: the size of interval to project to
        min_bin_size: the minimum bin size
    Returns:
        Tuple of the normalized bin_widths and their locations shape (..., num_bins) and (..., num_bins + 1)
    """
    num_bins = bins.shape[-1]
    # shape batch_shape + (dim)
    norm_bins = min_bin_size + (1 - min_bin_size * num_bins) * F.softmax(bins, dim=-1)
    cum_norm_bins = F.pad(torch.cumsum(norm_bins, dim=-1), pad=(1, 0), mode="constant", value=0.0)
    cum_norm_bins = tail_bound * (2 * cum_norm_bins - 1)
    norm_bins = torch.diff(cum_norm_bins)
    return norm_bins, cum_norm_bins
