import os

import pytest
import torch
from tensordict import TensorDict

from causica.datasets.causica_dataset_format import CAUSICA_DATASETS_PATH, DataEnum, load_data
from causica.datasets.tensordict_utils import tensordict_shapes
from causica.datasets.variable_types import VariableTypeEnum
from causica.distributions import ContinuousNoiseDist, JointNoiseModule, create_noise_modules
from causica.functional_relationships.linear_functional_relationships import LinearFunctionalRelationships
from causica.sem.distribution_parameters_sem import DistributionParametersSEM
from causica.training.evaluation import eval_ate_rmse, eval_intervention_likelihoods, eval_ite_rmse


def sem(data_dir: str) -> DistributionParametersSEM:
    graph = load_data(data_dir, DataEnum.TRUE_ADJACENCY).to(torch.float32)
    assert isinstance(graph, torch.Tensor)
    variables_metadata = load_data(data_dir, DataEnum.VARIABLES_JSON)
    train_data = load_data(data_dir, DataEnum.TRAIN, variables_metadata)
    assert isinstance(train_data, TensorDict)

    shapes: dict = tensordict_shapes(train_data)
    total_dim = sum(size[-1] for size in shapes.values())

    coef_matrix = torch.rand((total_dim, total_dim))

    func = LinearFunctionalRelationships(shapes, coef_matrix)
    noise_dist = JointNoiseModule(
        create_noise_modules(
            shapes=shapes,
            types=dict.fromkeys(shapes, VariableTypeEnum.CONTINUOUS),
            continuous_noise_dist=ContinuousNoiseDist.GAUSSIAN,
        )
    )
    return DistributionParametersSEM(graph=graph, noise_dist=noise_dist, func=func)


@pytest.mark.parametrize("dataset", ["csuite_linexp_2", "csuite_cts_to_cat"])
def test_evaluation_interventional_likelihood(dataset):
    root = os.path.join(CAUSICA_DATASETS_PATH, dataset)
    data = load_data(root, DataEnum.INTERVENTIONS)
    lik = eval_intervention_likelihoods([sem(root)], data[0])
    # this is double as there are two interventional datasets in the data
    assert lik.shape == torch.Size([data[0][0].intervention_data.shape[0] * 2])


@pytest.mark.parametrize("dataset", ["csuite_linexp_2", "csuite_cts_to_cat"])
def test_evaluation_ate(dataset):
    root = os.path.join(CAUSICA_DATASETS_PATH, dataset)
    ate = eval_ate_rmse([sem(root)], load_data(root, DataEnum.INTERVENTIONS)[0])
    assert ate.shape == tuple()


@pytest.mark.parametrize("dataset", ["csuite_linexp_2"])  # not testing csuite_cts_to_cat as cf file missing
def test_evaluation_ite(dataset):
    root = os.path.join(CAUSICA_DATASETS_PATH, dataset)
    ite = eval_ite_rmse([sem(root)], load_data(root, DataEnum.COUNTERFACTUALS)[0])
    assert ite.shape == tuple()
