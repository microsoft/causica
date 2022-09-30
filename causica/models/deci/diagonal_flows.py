import torch
from pyro.distributions import constraints
from pyro.distributions.torch_transform import TransformModule
from torch import nn

from ...utils.splines import unconstrained_RQS


class AffineDiagonalPyro(TransformModule):
    """
    This creates a diagonal affine transformation compatible with pyro transforms
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self, input_dim: int):
        super().__init__(cache_size=1)
        self.dim = input_dim
        self.a = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(input_dim), requires_grad=True)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        Args:
            x: tensor with shape [batch, input_dim]

        Returns:
            Transformed inputs
        """
        return self.a.exp().unsqueeze(0) * x + self.b.unsqueeze(0)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Reverse method
        Args:
            y: tensor with shape [batch, input]

        Returns:
            Reversed input
        """
        return (-self.a).exp().unsqueeze(0) * (y - self.b.unsqueeze(0))

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _ = x, y
        return self.a.unsqueeze(0)


def create_diagonal_spline_flow(flow_steps, features, num_bins=8, tail_bound=3):
    """
    Generate a composite flow as a sequence of diagonal Affine-Spline transofrmations. A final affine layer is appended to the end of transform.
    """
    return CompositeTransform(
        [
            CompositeTransform(
                [Affine_diagonal(features), PiecewiseRationalQuadraticTransform(features, num_bins, tail_bound)]
            )
            for i in range(flow_steps)
        ]
        + [Affine_diagonal(features)]
    )


class CompositeTransform(nn.Module):
    """Composes several transforms into one, in the order they are given. Provides forward and inverse methods.

    Args:
    transforms, List of transforms to compose.
    """

    def __init__(self, transforms):
        """Constructor.
        Args:
            transforms: an iterable of `Transform` objects.
        """
        super().__init__()
        self._transforms = nn.ModuleList(transforms)

    @staticmethod
    def _cascade(inputs, funcs):
        """
        Sequentially apply all transofrmations in forward or reverse mode and accumulate dimensionwise log determinant
        """
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size, inputs.shape[1])
        for func in funcs:
            outputs, logabsdet = func(outputs)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(self, inputs):
        """
        Args:
            input: (batch_size, input_dim)
        Returns:
            transformed_input, jacobian_log_determinant: (batch_size, input_dim), (batch_size, input_dim)
        """
        funcs = self._transforms
        return self._cascade(inputs, funcs)

    def inverse(self, inputs):
        """
        Args:
            input: (batch_size, input_dim)
        Returns:
            transformed_input, jacobian_log_determinant: (batch_size, input_dim), (batch_size, input_dim)
        """
        funcs = (transform.inverse for transform in self._transforms[::-1])
        return self._cascade(inputs, funcs)


class Affine_diagonal(nn.Module):
    """
    Layer that implements transofrmation ax + b. All dimensions of x are treated as independent.
    """

    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.a = nn.Parameter(torch.ones(dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(dim), requires_grad=True)

    def forward(self, inputs):
        """
        Args:
            input: (batch_size, input_dim)
        Returns:
            transformed_input, jacobian_log_determinant: (batch_size, input_dim), (batch_size, input_dim)
        """
        return self.a.exp().unsqueeze(0) * inputs + self.b.unsqueeze(0), self.a.unsqueeze(0)

    def inverse(self, inputs):
        """
        Args:
            input: (batch_size, input_dim)
        Returns:
            transformed_input, jacobian_log_determinant: (batch_size, input_dim), (batch_size, input_dim)
        """
        return (-self.a).exp().unsqueeze(0) * (inputs - self.b.unsqueeze(0)), -self.a.unsqueeze(0)


class PiecewiseRationalQuadraticTransform(nn.Module):
    """
    Layer that implements a spline-cdf (https://arxiv.org/abs/1906.04032) transformation.
     All dimensions of x are treated as independent, no coupling is used. This is needed
    to ensure invertibility in our additive noise SEM.

    Args:
        dim: dimensionality of input,
        num_bins: how many bins to use in spline,
        tail_bound: distance of edgemost bins relative to 0,
        init_scale: standar deviation of Gaussian from which spline parameters are initialised
    """

    def __init__(
        self,
        dim,
        num_bins=8,
        tail_bound=3.0,
        init_scale=1e-2,
    ):
        super().__init__()

        self.dim = dim
        self.num_bins = num_bins
        self.min_bin_width = 1e-3
        self.min_bin_height = 1e-3
        self.min_derivative = 1e-3
        self.tail_bound = tail_bound
        self.init_scale = init_scale

        self.params = nn.Parameter(self.init_scale * torch.randn(self.dim, self.num_bins * 3 - 1), requires_grad=True)

    def _piecewise_cdf(self, inputs, inverse=False):
        params_batch = self.params.unsqueeze(dim=(0)).expand(inputs.shape[0], -1, -1)

        unnormalized_widths = params_batch[..., : self.num_bins]
        unnormalized_heights = params_batch[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = params_batch[..., 2 * self.num_bins :]

        return unconstrained_RQS(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            tail_bound=self.tail_bound,
        )

    def forward(self, inputs):
        """
        Args:
            input: (batch_size, input_dim)
        Returns:
            transformed_input, jacobian_log_determinant: (batch_size, input_dim), (batch_size, input_dim)
        """
        return self._piecewise_cdf(inputs, inverse=False)

    def inverse(self, inputs):
        """
        Args:
            input: (batch_size, input_dim)
        Returns:
            transformed_input, jacobian_log_determinant: (batch_size, input_dim), (batch_size, input_dim)
        """
        return self._piecewise_cdf(inputs, inverse=True)
