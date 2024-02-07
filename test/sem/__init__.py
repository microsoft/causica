import torch

from causica.distributions import JointNoiseModule, Noise, NoiseModule, UnivariateNormalNoiseModule
from causica.functional_relationships import LinearFunctionalRelationships, RFFFunctionalRelationships
from causica.sem.distribution_parameters_sem import DistributionParametersSEM


def create_lingauss_sem(
    shapes: dict[str, torch.Size],
    coef_matrix: torch.Tensor,
    graph: torch.Tensor,
    log_scale: float = 0.0,
) -> DistributionParametersSEM:
    """Creates a dummy SEM with linear functional relationships and Gaussian noise."""
    independent_noise_modules = {
        name: UnivariateNormalNoiseModule(shape[-1], init_log_scale=log_scale) for name, shape in shapes.items()
    }
    noise_dist = JointNoiseModule(independent_noise_modules)
    func = LinearFunctionalRelationships(shapes, coef_matrix)
    return DistributionParametersSEM(graph=graph, noise_dist=noise_dist, func=func)


def create_rffgauss_sem(
    shapes: dict[str, torch.Size],
    random_features: torch.Tensor,
    coeff_matrix: torch.Tensor,
    graph: torch.Tensor,
    log_scale: float = 0.0,
) -> DistributionParametersSEM:
    """Creates a dummy SEM with RFF functional relationships and Gaussian noise."""
    independent_noise_modules = {
        name: UnivariateNormalNoiseModule(shape[-1], init_log_scale=log_scale) for name, shape in shapes.items()
    }
    noise_dist = JointNoiseModule(independent_noise_modules)
    func = RFFFunctionalRelationships(shapes, random_features, coeff_matrix)
    return DistributionParametersSEM(graph=graph, noise_dist=noise_dist, func=func)
