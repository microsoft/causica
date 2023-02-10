from typing import Callable

import torch.distributions as td
from torch import nn


class ParametrizedDistribution(nn.Module):
    """
    A container class for a distribution and its parameters.

    In PyTorch, `Distribution`s are not learnable as the parameters are inputs to the class.
    This can mean that after updates, if any other values within the distribution are derived from the parameters,
    they may not be updated.

    This simple wrapper stores a distribution class and its parameters (as a dict) together.
    It dynamically instantiates the distribution when required. It is assumed that distribution instantiation is cheap.
    For the reason above, the dynamically instantiated distribution should be persisted for as little time as possible (particularly if parameters are updated).

    For example:
    ```
    loc = Parameter(0., requires_grad=True)
    scale = Parameter(1., requires_grad=True)
    paramdist = ParametrizedDistribution(
        torch.distributions.Normal,
        torch.nn.ParameterDict(dict(loc=loc, scale=scale)),
    )
    ```
    And then to use:
    ```
    dist: torch.distributions.Normal = paramdist.forward()
    mean = dist.mean
    ```
    """

    def __init__(self, distribution_class: Callable[..., td.Distribution], param_dict: nn.ParameterDict, **kwargs):
        """
        Args:
            distribution_class: the class of the distribution that will be instantiated, e.g. torch.distributions.Normal
            param_dict: the dictionary of parameters to be passed to the distribution on instantiation, e.g. {"loc": ..., "scale": ...}
            **kwargs: Any other arguments to be passed to the distribution on instantiation (that aren't parameters), e.g. `validate_args=True`
        """
        super().__init__()
        self.param_dict = param_dict
        self.distribution_class = distribution_class
        self.kwargs = kwargs

    def forward(self) -> td.Distribution:
        """Return the instantiated `Distribution`"""
        return self.distribution_class(**dict(self.param_dict.items()), **self.kwargs)
