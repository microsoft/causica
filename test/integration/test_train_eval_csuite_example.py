import mlflow
import pytest
from examples.csuite_example.eval_csuite_example import eval_csuite
from examples.csuite_example.train_csuite_example import AugLagLRConfig, ICGNNConfig, TrainingConfig, train

from causica.training.trainable_container import NoiseDist


@pytest.mark.parametrize(
    "dataset, noise_dist",
    [
        ("csuite_linexp_2", NoiseDist.GAUSSIAN),
        ("csuite_cat_to_cts", NoiseDist.SPLINE),
        ("csuite_large_backdoor_binary_t", NoiseDist.GAUSSIAN),
        ("csuite_mixed_simpson", NoiseDist.GAUSSIAN),
    ],
)
def test_train_eval_csuite_no_crash(dataset, noise_dist):
    """Test that PyTorch Lightning runs on a multidimensional dataset."""

    icgnn_config = ICGNNConfig()
    training_config = TrainingConfig(max_epoch=1, noise_dist=noise_dist)
    auglag_config = AugLagLRConfig()

    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow.end_run()
    with mlflow.start_run():
        container = train(
            dataset=dataset,
            seed=123,
            auglag_config=auglag_config,
            icgnn_config=icgnn_config,
            training_config=training_config,
        )
        eval_csuite(container, ite=False)
