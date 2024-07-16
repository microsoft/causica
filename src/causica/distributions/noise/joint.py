from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Optional, TypeVar

import torch
from tensordict import TensorDict
from torch import nn

from causica.datasets.variable_types import VariableTypeEnum
from causica.distributions.noise.bernoulli import BernoulliNoiseModule
from causica.distributions.noise.categorical import CategoricalNoiseModule
from causica.distributions.noise.noise import Noise, NoiseModule
from causica.distributions.noise.spline import SplineNoiseModule
from causica.distributions.noise.univariate_normal import UnivariateNormalNoiseModule

SelectionType = TypeVar("SelectionType", str, Iterable[str])


class JointNoise(Noise[TensorDict]):
    """Represents an independent joint noise distribution of multiple variables types.

    Samples are TensorDicts containining independent variables.
    """

    arg_constraints = {}

    def __init__(self, independent_noise_dists: Mapping[str, Noise[torch.Tensor]]):
        shapes = defaultdict[torch.Size, list[str]](list)
        for name, noise_dist in independent_noise_dists.items():
            shapes[noise_dist.batch_shape].append(name)
        if len(shapes) > 1:
            raise ValueError(f"Incompatible batch shapes: {shapes}")

        batch_shape = next(iter(shapes)) if shapes else torch.Size()
        super().__init__(batch_shape=batch_shape, event_shape=torch.Size([len(independent_noise_dists)]))
        self._independent_noise_dists = dict(independent_noise_dists)

    def _apply_individually(self, value: TensorDict, func: Callable[[Noise, torch.Tensor], torch.Tensor]) -> TensorDict:
        return TensorDict(
            {name: func(noise_dist, value.get(name)) for name, noise_dist in self._independent_noise_dists.items()},
            batch_size=value.batch_size,
            device=value.device,
        )

    def sample_to_noise(self, samples: TensorDict) -> TensorDict:
        return self._apply_individually(samples, lambda noise_dist, x: noise_dist.sample_to_noise(x))

    def noise_to_sample(self, noise: TensorDict) -> TensorDict:
        return self._apply_individually(noise, lambda noise_dist, x: noise_dist.noise_to_sample(x))

    def sample(self, sample_shape: torch.Size = torch.Size()) -> TensorDict:
        return TensorDict(
            {name: noise_dist.sample(sample_shape) for name, noise_dist in self._independent_noise_dists.items()},
            batch_size=sample_shape + self.batch_shape,
        )

    def log_prob(self, value: TensorDict) -> torch.Tensor:
        """Compute the log probs of the given values.

        Note:
            Produces `log_probs` of the same shape as the inner noise distributions.

        Args:
            value: Values matching the inner independent noise distributions by key. The batch shape of the TensorDict
                   should be set to catch all dims except those associated with the event shape for the expected return
                   shape to behave as specified.

        Returns:
            torch.Tensor: The log probability of the given value, of shape
                           `torch.broadcast_shapes(value.batch_size, self.batch_shape)`.
        """
        log_probs = [noise_dist.log_prob(value[name]) for name, noise_dist in self._independent_noise_dists.items()]
        return torch.sum(torch.stack(log_probs, dim=0), dim=0)

    @property
    def support(self) -> dict[str, Optional[Any]]:
        return {name: noise_dist.support for name, noise_dist in self._independent_noise_dists.items()}

    @property
    def mode(self) -> TensorDict:
        return TensorDict(
            {name: noise_dist.mode for name, noise_dist in self._independent_noise_dists.items()},
            batch_size=self.batch_shape,
        )

    @property
    def mean(self) -> TensorDict:
        return TensorDict(
            {name: noise_dist.mean for name, noise_dist in self._independent_noise_dists.items()},
            batch_size=self.batch_shape,
        )

    def entropy(self) -> torch.Tensor:
        entropies = [noise_dist.entropy() for noise_dist in self._independent_noise_dists.values()]
        return torch.sum(torch.stack(entropies, dim=0), dim=0)


class ContinuousNoiseDist(Enum):
    SPLINE = "spline"
    GAUSSIAN = "gaussian"


def create_noise_modules(
    shapes: dict[str, torch.Size],
    types: dict[str, VariableTypeEnum],
    continuous_noise_dist: ContinuousNoiseDist,
) -> dict[str, NoiseModule[Noise[torch.Tensor]]]:
    """Create noise modules for each item of shapes and types.

    Args:
        shapes: The shape of each distribution. Currently only the last dimension is used.
        types: Names of variables mapping to the variable type `VariableTypeEnum`.
        continuous_noise_dist: The noise module to use for variable types of `VariableTypeEnum.CONTINUOUS`.

    Raises:
        ValueError: If any of the types or the continuous noise distribution is incorrectly specified.

    Returns
        Dict of independent noise modules following the shape and type specifications.
    """
    noise_modules: dict[str, NoiseModule] = {}
    for key, shape in shapes.items():
        size = shape[-1]
        var_type = types[key]

        noise_module: NoiseModule
        if var_type == VariableTypeEnum.CATEGORICAL:
            noise_module = CategoricalNoiseModule(size)
        elif var_type == VariableTypeEnum.BINARY:
            noise_module = BernoulliNoiseModule(size)
        elif var_type == VariableTypeEnum.CONTINUOUS:
            if continuous_noise_dist == ContinuousNoiseDist.SPLINE:
                noise_module = SplineNoiseModule(size)
            elif continuous_noise_dist == ContinuousNoiseDist.GAUSSIAN:
                noise_module = UnivariateNormalNoiseModule(size)
            else:
                raise ValueError(f"Invalid continuous noise distribution {continuous_noise_dist}.")
        else:
            raise ValueError(f"Invalid variable type {var_type}")
        noise_modules[key] = noise_module
    return noise_modules


class JointNoiseModule(NoiseModule[JointNoise]):
    """Represents JointNoise with learnable parameters.

    Each noise module is used independently on their corresponding key of sample TensorDicts.
    """

    def __init__(self, independent_noise_modules: Mapping[str, NoiseModule[Noise[torch.Tensor]]]):
        """
        Args:
            independent_noise_modules: Noise modules to be applied keywise to input TensorDicts. Could e.g. be created
                                       by `create_noise_distributions`.
        """
        super().__init__()
        self.noise_modules = nn.ModuleDict(independent_noise_modules)

    def forward(self, x: Optional[tuple[TensorDict, TensorDict] | TensorDict] = None) -> JointNoise:
        """
        Some noise_module allows to access to tuple of two Tensors rather than a single Tensor (e.g. univariate_normal, univariate_laplace, univariate_cauchy)
        Note that if a tuple is provided to a noise_module that does not allow tuple of tensor, this forward call will raise an error.
        """

        if isinstance(x, tuple):
            x, y = x
            noise_distributions = {
                name: noise_module((x.get(name), y.get(name))) for name, noise_module in self.noise_modules.items()
            }
        elif isinstance(x, TensorDict):
            noise_distributions = {name: noise_module(x.get(name)) for name, noise_module in self.noise_modules.items()}
        else:
            noise_distributions = {name: noise_module() for name, noise_module in self.noise_modules.items()}

        return JointNoise(independent_noise_dists=noise_distributions)

    def __getitem__(self, selection: Iterable[str]) -> "JointNoiseModule":
        """Return a JointNoiseModule representing the subset of variables specified in selection."""
        selected_independent_noise_modules = {name: self.noise_modules[name] for name in selection}
        # ModuleDict doesn't allow tracking value types, so ignore the type here.
        return JointNoiseModule(selected_independent_noise_modules)  # type: ignore

    def keys(self) -> tuple[str, ...]:
        """The keys for the different noise modules in order."""
        return tuple(self.noise_modules)
