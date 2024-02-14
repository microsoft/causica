from __future__ import annotations

from typing import Optional

import torch
import torch.distributions as td

from causica.data_generation.samplers.sampler import Sampler
from causica.functional_relationships.functional_relationships import FunctionalRelationships
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
    """Sample a Non Linear Functional Relationship, by providing two distributions:
    a first distribution for the random features, and a second distribution for the linear outer coefficients."""

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
