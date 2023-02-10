import numpy as np
import pytest
import torch
from tensordict import TensorDict

from causica.datasets.csuite_data import CSUITE_DATASETS_PATH, DataEnum, load_data
from causica.datasets.tensordict_utils import tensordict_shapes
from causica.distributions import NoiseAccessibleMultivariateNormal
from causica.functional_relationships.linear_functional_relationships import LinearFunctionalRelationships
from causica.sem.distribution_parameters_sem import DistributionParametersSEM
from causica.training.evaluation import eval_ate_rmse, eval_intervention_likelihoods, eval_ite_rmse

DATASET = "csuite_linexp_2"


# pylint: disable=redefined-outer-name
def sem(dataset):
    graph: np.ndarray = torch.tensor(load_data(CSUITE_DATASETS_PATH, dataset, DataEnum.TRUE_ADJACENCY)).to(
        torch.float32
    )
    train_data: TensorDict = load_data(CSUITE_DATASETS_PATH, dataset, DataEnum.TRAIN)

    shapes = tensordict_shapes(train_data)
    total_dim = sum(size[-1] for size in shapes.values())

    coef_matrix = torch.rand((total_dim, total_dim))

    func = LinearFunctionalRelationships(shapes, coef_matrix)
    # create new noise dists for each node
    noise_dist = {
        key: (
            lambda x, shape=shape: NoiseAccessibleMultivariateNormal(
                x, covariance_matrix=torch.diag_embed(1e-10 * torch.ones(shape))
            )
        )
        for key, shape in shapes.items()
    }
    return DistributionParametersSEM(graph=graph, node_names=train_data.keys(), noise_dist=noise_dist, func=func)


@pytest.mark.parametrize("dataset", ["csuite_linexp_2", "csuite_cts_to_cat"])
def test_evaluation_int_lik(dataset):
    lik = eval_intervention_likelihoods(
        [sem(dataset)], load_data(CSUITE_DATASETS_PATH, dataset, DataEnum.INTERVENTIONS)
    )
    assert lik.shape == tuple()


@pytest.mark.parametrize("dataset", ["csuite_linexp_2", "csuite_cts_to_cat"])
def test_evaluation_ate(dataset):
    ate = eval_ate_rmse([sem(dataset)], load_data(CSUITE_DATASETS_PATH, dataset, DataEnum.INTERVENTIONS))
    assert ate.shape == tuple()


@pytest.mark.parametrize("dataset", ["csuite_linexp_2"])  # not testing csuite_cts_to_cat as cf file missing
def test_evaluation_ite(dataset):
    ite = eval_ite_rmse([sem(dataset)], load_data(CSUITE_DATASETS_PATH, DATASET, DataEnum.COUNTERFACTUALS))
    assert ite.shape == tuple()
