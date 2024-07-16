from __future__ import annotations

from typing import Optional

import torch
import torch.distributions as td

from causica.data_generation.samplers.sampler import Sampler
from causica.functional_relationships.functional_relationships import FunctionalRelationships
from causica.functional_relationships.heteroscedastic_rff_functional_relationships import (
    HeteroscedasticRFFFunctionalRelationships,
)
from causica.functional_relationships.linear_functional_relationships import LinearFunctionalRelationships
from causica.functional_relationships.rff_functional_relationships import RFFFunctionalRelationships


class FunctionalRelationshipsSampler(Sampler[FunctionalRelationships]):
    """Abstract class for sampling Functional Relationships."""

    def __init__(self, shapes_dict: dict[str, torch.Size]) -> None:
        super().__init__()

        self.shapes_dict = shapes_dict


class LinearRelationshipsSampler(FunctionalRelationshipsSampler):
    """Sample a Linear Functional Relationship, by providing a distribution for the coefficient matrix."""

    def __init__(
        self,
        scale_dist: td.Distribution,
        shapes_dict: dict[str, torch.Size],
        bias_dist: Optional[td.Distribution] = None,
    ):
        super().__init__(shapes_dict)
        self.scale_dist = scale_dist
        self.bias_dist = bias_dist

    def sample(self) -> FunctionalRelationships:
        sample_bias = self.bias_dist.sample() if self.bias_dist is not None else None
        return LinearFunctionalRelationships(
            shapes=self.shapes_dict,
            initial_linear_coefficient_matrix=self.scale_dist.sample(),
            initial_bias=sample_bias,
        )


class RFFFunctionalRelationshipsSampler(FunctionalRelationshipsSampler):
    """Sample a Non Linear Functional Relationship, by providing two distributions.

    The sampler uses two distributions:
        - a first distribution for the random features
        - a second distribution for the linear outer coefficients."""

    def __init__(
        self,
        rf_dist: td.Distribution,
        coeff_dist: td.Distribution,
        shapes_dict: dict[str, torch.Size],
        bias_dist: Optional[td.Distribution] = None,
        length_dist: Optional[td.Distribution] = None,
        out_dist: Optional[td.Distribution] = None,
        angle_dist: Optional[td.Distribution] = None,
    ):
        super().__init__(shapes_dict)
        self.bias_dist = bias_dist
        self.length_dist = length_dist
        self.out_dist = out_dist
        self.angle_dist = angle_dist
        self.rf_dist = rf_dist
        self.coeff_dist = coeff_dist

    def sample(self) -> FunctionalRelationships:
        sample_bias = self.bias_dist.sample() if self.bias_dist is not None else None
        sample_length = self.length_dist.sample() if self.length_dist is not None else None
        sample_out = self.out_dist.sample() if self.out_dist is not None else None
        sample_angle = self.angle_dist.sample() if self.angle_dist is not None else None

        return RFFFunctionalRelationships(
            shapes=self.shapes_dict,
            initial_random_features=self.rf_dist.sample(),
            initial_coefficients=self.coeff_dist.sample(),
            initial_bias=sample_bias,
            initial_length_scales=sample_length,
            initial_output_scales=sample_out,
            initial_angles=sample_angle,
        )


class HeteroscedasticRFFFunctionalRelationshipsSampler(FunctionalRelationshipsSampler):
    """Sample a Post Non Linear Functional Relationship, by providing two distributions.

    The sampler uses two distributions:
        - a first distribution for the random features
        - a second distribution for the linear outer coefficients.
    The model is of the form: x = log(1+ exp(g(x))) where g is the RFF model."""

    def __init__(
        self,
        rf_dist: td.Distribution,
        coeff_dist: td.Distribution,
        shapes_dict: dict[str, torch.Size],
        bias_dist: Optional[td.Distribution] = None,
        length_dist: Optional[td.Distribution] | Optional[float] = None,
        out_dist: Optional[td.Distribution] | Optional[float] = None,
        angle_dist: Optional[td.Distribution] = None,
        log_scale: bool = False,
    ):
        """
        Args:
            rf_dist: Distribution for the random features
            coeff_dist: Distribution for the linear outer coefficients
            shapes_dict: Shapes of the different variables
            bias_dist: Distribution for the bias
            length_dist: Distribution for the length scales
            out_dist: Distribution for the output scales
            angle_dist: Distribution for the angles
            log_scale: If True, the output is log(log(1+exp(g(x)))), else it is log(1+exp(g(x)))
        """

        super().__init__(shapes_dict)
        self.bias_dist = bias_dist
        self.length_dist = length_dist
        self.out_dist = out_dist
        self.angle_dist = angle_dist
        self.rf_dist = rf_dist
        self.coeff_dist = coeff_dist
        self.log_scale = log_scale

    def sample(self) -> HeteroscedasticRFFFunctionalRelationships:
        sample_coeff = self.coeff_dist.sample()
        sample_rf = self.rf_dist.sample()
        sample_bias = self.bias_dist.sample() if self.bias_dist is not None else None
        sample_angle = self.angle_dist.sample() if self.angle_dist is not None else None

        if self.length_dist is not None:
            if isinstance(self.length_dist, td.Distribution):
                sample_length = self.length_dist.sample()
            else:
                sample_length = self.length_dist * torch.ones_like(sample_rf[0])
        else:
            sample_length = None

        if self.out_dist is not None:
            if isinstance(self.out_dist, td.Distribution):
                sample_out = self.out_dist.sample()
            else:
                sample_out = self.out_dist * torch.ones_like(sample_rf[0])
        else:
            sample_out = None

        return HeteroscedasticRFFFunctionalRelationships(
            shapes=self.shapes_dict,
            initial_random_features=sample_rf,
            initial_coefficients=sample_coeff,
            initial_bias=sample_bias,
            initial_length_scales=sample_length,
            initial_output_scales=sample_out,
            initial_angles=sample_angle,
            log_scale=self.log_scale,
        )
