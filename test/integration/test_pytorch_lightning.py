import mlflow
import pytest
import pytorch_lightning as pl
import torch
from examples.csuite_example.eval_csuite_example import eval_csuite

from causica.datasets.csuite_data import DataEnum, get_csuite_path, load_data
from causica.distributions import ExpertGraphContainer
from causica.lightning.data_modules import CSuiteDataModule
from causica.lightning.modules import DECIModule


def _module_to_parameter(module: DECIModule):
    """Get some parameter from the module"""
    return module.optimizers().optimizer.param_groups[0]["params"][1]


@pytest.mark.parametrize("dataset", ["csuite_mixed_simpson"])
def test_pytorch_lightning_deterministic(dataset):
    """An integration test to test that PyTorch Lightning runs deterministically on datasets."""
    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()

    trainer = pl.Trainer(fast_dev_run=True)
    module = DECIModule()
    data_module = CSuiteDataModule(dataset_name=dataset)
    trainer.fit(module, data_module)

    module2 = DECIModule()
    trainer.fit(module2, data_module)
    # check that some parameter in the model is equal after training, indictating determinism
    torch.testing.assert_close(_module_to_parameter(module), _module_to_parameter(module2))

    eval_csuite(module.container, ite=False)


@pytest.mark.parametrize("dataset", ["csuite_mixed_simpson"])
def test_pytorch_lightning_expert_input(dataset):
    """Test that the additional expert graph and constraints can be used."""
    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()

    trainer = pl.Trainer(fast_dev_run=True)
    data_module = CSuiteDataModule(dataset_name=dataset)
    adj_path = get_csuite_path(CSuiteDataModule.DEFAULT_CSUITE_PATH, dataset, DataEnum.TRUE_ADJACENCY)
    adj_matrix = torch.tensor(load_data(CSuiteDataModule.DEFAULT_CSUITE_PATH, dataset, DataEnum.TRUE_ADJACENCY))
    expert_graph_container = ExpertGraphContainer(
        dag=adj_matrix, mask=torch.ones_like(adj_matrix), confidence=0.9, scale=1.0
    )
    module = DECIModule(constraint_matrix_path=adj_path, expert_graph_container=expert_graph_container)
    trainer.fit(module, data_module)

    learned_graph = module.container.vardist().sample(torch.Size([]))
    torch.testing.assert_close(learned_graph, expert_graph_container.dag.to(dtype=torch.float32))
