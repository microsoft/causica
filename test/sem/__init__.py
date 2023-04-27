import torch

from causica.distributions import JointNoiseModule, Noise, NoiseModule, UnivariateNormalNoiseModule
from causica.functional_relationships import LinearFunctionalRelationships
from causica.sem.distribution_parameters_sem import DistributionParametersSEM


def create_lingauss_sem(
    shapes: dict[str, torch.Size],
    coef_matrix: torch.Tensor,
    graph: torch.Tensor,
    log_scale: float = 0.0,
) -> DistributionParametersSEM:
    independent_noise_modules: dict[str, NoiseModule[Noise[torch.Tensor]]] = {
        name: UnivariateNormalNoiseModule(shape[-1], init_log_scale=log_scale) for name, shape in shapes.items()
    }
    noise_dist = JointNoiseModule(independent_noise_modules)
    func = LinearFunctionalRelationships(shapes, coef_matrix)
    return DistributionParametersSEM(graph=graph, noise_dist=noise_dist, func=func)
