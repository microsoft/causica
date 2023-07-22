import mlflow
import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from causica.distributions import ExpertGraphContainer
from causica.distributions.noise.joint import ContinuousNoiseDist
from causica.lightning.data_modules.variable_spec_data import CSuiteDataModule
from causica.lightning.modules.deci_module import DECIModule


def _module_to_parameter(module: DECIModule):
    """Get some parameter from the module"""
    optimizer = module.optimizers()
    if isinstance(optimizer, torch.optim.Optimizer):
        return optimizer.param_groups[0]["params"][1]
    raise ValueError("Only expected one optimizer")


@pytest.mark.parametrize(
    "dataset, noise_dist",
    [
        ("csuite_linexp_2", ContinuousNoiseDist.GAUSSIAN),
        ("csuite_cat_to_cts", ContinuousNoiseDist.SPLINE),
        ("csuite_large_backdoor_binary_t", ContinuousNoiseDist.GAUSSIAN),
        ("csuite_mixed_simpson", ContinuousNoiseDist.GAUSSIAN),
    ],
)
def test_pytorch_lightning_deterministic(dataset, noise_dist):
    """An integration test to test that PyTorch Lightning runs deterministically on datasets."""
    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()

    trainer = pl.Trainer(fast_dev_run=True)
    module = DECIModule(noise_dist=noise_dist)
    data_module = CSuiteDataModule(dataset_name=dataset)
    trainer.fit(module, data_module)

    module2 = DECIModule()
    trainer.fit(module2, data_module)
    # check that some parameter in the model is equal after training, hence deterministic
    torch.testing.assert_close(_module_to_parameter(module), _module_to_parameter(module2))

    trainer.test(module, data_module)


@pytest.mark.parametrize("dataset", ["csuite_mixed_simpson"])
def test_pytorch_lightning_expert_input(tmp_path, dataset):
    """Test that the additional expert graph and constraints can be used."""
    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()

    trainer = pl.Trainer(fast_dev_run=True)
    data_module = CSuiteDataModule(dataset_name=dataset)

    data_module.prepare_data()
    adj_matrix = data_module.true_adj
    expert_graph_container = ExpertGraphContainer(
        dag=adj_matrix, mask=torch.ones_like(adj_matrix), confidence=0.9, scale=1.0
    )

    constraint_matrix_path = tmp_path / "constraint_graph.npy"
    with constraint_matrix_path.open("wb") as f:
        np.save(f, adj_matrix.numpy())

    module = DECIModule(
        constraint_matrix_path=str(constraint_matrix_path), expert_graph_container=expert_graph_container
    )
    trainer.fit(module, data_module)

    learned_sem, *_ = module.sem_module().sample(torch.Size([]))
    torch.testing.assert_close(learned_sem.graph, expert_graph_container.dag.to(dtype=torch.float32))
