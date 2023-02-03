from __future__ import annotations

from .ddeci import ADMGParameterisedDDECI, BowFreeDDECI


class ADMGParameterisedDDECIGaussian(ADMGParameterisedDDECI):
    """ADMGParameterisedDDECI where the additive noise SEM base distribution is fixed to being a Gaussian."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, base_distribution_type="gaussian", **kwargs)

    @classmethod
    def name(cls) -> str:
        return "admg_ddeci_gaussian"


class BowFreeDDECIGaussian(BowFreeDDECI):
    """BowFreeDDECI where the additive noise SEM base distribution is fixed to being a Gaussian."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, base_distribution_type="gaussian", **kwargs)

    @classmethod
    def name(cls) -> str:
        return "bowfree_ddeci_gaussian"
