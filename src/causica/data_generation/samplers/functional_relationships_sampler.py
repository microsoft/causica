from __future__ import annotations

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

    def __init__(self, scale_dist: td.Distribution, shapes_dict: dict[str, torch.Size]):
        super().__init__(shapes_dict)
        self.scale_dist = scale_dist

    def sample(self) -> FunctionalRelationships:
        return LinearFunctionalRelationships(
            shapes=self.shapes_dict, initial_linear_coefficient_matrix=self.scale_dist.sample()
        )


class RFFFunctionalRelationshipsSampler(FunctionalRelationshipsSampler):
    """Sample a Non Linear Functional Relationship, by providing two distributions:
    a first distribution for the random features, and a second distribution for the linear outer coefficients."""

    def __init__(self, rf_dist: td.Distribution, coeff_dist: td.Distribution, shapes_dict: dict[str, torch.Size]):
        super().__init__(shapes_dict)
        self.rf_dist = rf_dist
        self.coeff_dist = coeff_dist

    def sample(self) -> FunctionalRelationships:
        return RFFFunctionalRelationships(
            shapes=self.shapes_dict,
            initial_random_features=self.rf_dist.sample(),
            initial_coefficients=self.coeff_dist.sample(),
        )
