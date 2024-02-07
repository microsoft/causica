"""
Implementation of the Piecewise Rational Quadratic Transform

This is pretty much a copy-paste of
    https://github.com/tonyduan/normalizing-flows/blob/master/nf/utils.py

We should consider using the Pyro implementation.
"""

import torch
import torch.distributions as td

from causica.distributions.noise.spline.bayesiains_nsf_rqs import unconstrained_rational_quadratic_spline

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

    def _piecewise_cdf(self, inputs: torch.Tensor, inverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the Cumulative Density function at `inputs`

        Args:
            inputs: the positions at which to evaluate the cdf shape batch_shape + (input_dim)
            inverse: whether this is forwards or backwards transform
        Returns:
            input_evaluations and absolute log determinants, a tuple of tensors of shape batch_shape + (input_dim)
        """
        assert len(inputs.shape) > 1  # TODO(JJ) accept 1d inputs
        batch_shape = inputs.shape[:-1]
        # shape batch_shape + (dim, 3 * num_bins - 1)
        expanded_knot_locations = self.knot_locations[None, ...].expand(*batch_shape, -1, -1, -1)
        expanded_derivatives = self.derivatives[None, ...].expand(*batch_shape, -1, -1)

        return unconstrained_rational_quadratic_spline(
            inputs=inputs,
            unnormalized_widths=expanded_knot_locations[..., 0],
            unnormalized_heights=expanded_knot_locations[..., 1],
            unnormalized_derivatives=expanded_derivatives,
            inverse=inverse,
            tail_bound=self.tail_bound,
        )

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
