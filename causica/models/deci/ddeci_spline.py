from __future__ import annotations

from .ddeci import ADMGParameterisedDDECI, BowFreeDDECI


class ADMGParameterisedDDECISpline(ADMGParameterisedDDECI):
    """ADMGParameterisedDDECI where the additive noise SEM base distribution is fixed to being a learnable spline distribution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, base_distribution_type="spline", **kwargs)

    @classmethod
    def name(cls) -> str:
        return "admg_ddeci_spline"


class BowFreeDDECISpline(BowFreeDDECI):
    """BowFreeDDECI where the additive noise SEM base distribution is fixed to being a learnable spline distirbution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, base_distribution_type="spline", **kwargs)

    @classmethod
    def name(cls) -> str:
        return "bowfree_ddeci_spline"
