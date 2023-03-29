from typing import Generic, TypeVar

from torch import distributions as td
from torch import nn

DistributionType_co = TypeVar("DistributionType_co", bound=td.Distribution, covariant=True)


class DistributionModule(Generic[DistributionType_co], nn.Module):
    """Baseclass for modules returning distributions.

    Useful e.g. to create variational approximations of distributions.

    Subclasses are expected to implement a `forward` method that returns a concrete `td.Distribution` and should usually
    inherit from a conrete version of this class, i.e. `DistributionModule[<td.Distribution subclass>]`.
    """

    def __call__(self, *args, **kwargs) -> DistributionType_co:
        """Return a td.Distribution."""
        return super().__call__(*args, **kwargs)
