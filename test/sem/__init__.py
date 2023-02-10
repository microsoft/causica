from collections import OrderedDict
from typing import Callable, Dict

import torch

from causica.distributions import NoiseAccessibleMultivariateNormal
from causica.distributions.noise_accessible.noise_accessible import NoiseAccessibleDistribution
from causica.functional_relationships import LinearFunctionalRelationships
from causica.sem.distribution_parameters_sem import DistributionParametersSEM


def create_lingauss_sem(
    shapes: "OrderedDict[str, torch.Size]", coef_matrix: torch.Tensor, graph: torch.Tensor, scale: float = 1.0
) -> DistributionParametersSEM:
    def get_noise_accessible_constructor(shape: torch.Size) -> Callable[[torch.Tensor], NoiseAccessibleDistribution]:
        def constructor(x: torch.Tensor) -> NoiseAccessibleDistribution:
            return NoiseAccessibleMultivariateNormal(
                x, covariance_matrix=torch.diag_embed(scale * scale * torch.ones(shape))
            )

        return constructor

    func = LinearFunctionalRelationships(shapes, coef_matrix)
    # create new noise dists for each node
    noise_dist: Dict[str, Callable[[torch.Tensor], NoiseAccessibleDistribution]] = {
        key: get_noise_accessible_constructor(shape) for key, shape in shapes.items()
    }
    return DistributionParametersSEM(graph=graph, node_names=list(shapes.keys()), noise_dist=noise_dist, func=func)
