import os
from pathlib import Path

import mlflow
import pytest
import yaml
from jsonargparse import Namespace
from mlflow import ActiveRun
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.cli import LightningArgumentParser
from pytorch_lightning.trainer.states import TrainerFn

from causica.lightning.callbacks import MLFlowSaveConfigCallback


@pytest.fixture(name="local_mlflow_run", scope="function")
def local_mlflow_run_fn(tmp_path: Path) -> ActiveRun:
    """Return an mlflow run tracking to a temporary directory."""
    mlflow.set_tracking_uri(tmp_path)
    experiment_id = mlflow.create_experiment("experiment")
    return mlflow.start_run(experiment_id=experiment_id)


def test_mlflow_save_config_callback(local_mlflow_run: ActiveRun):
    """Test that the config is successfully saved to mlflow."""
    namespace = Namespace(foo="bar", nested={"baz": "foobar"})
    parser = LightningArgumentParser()
    for key in namespace.keys():
        parser.add_argument(f"--{key}")

    with local_mlflow_run:
        callback = MLFlowSaveConfigCallback(parser=parser, config=namespace, config_filename="config.yaml")
        callback.setup(Trainer(), LightningModule(), stage=TrainerFn.FITTING)

    base_artifact_uri = local_mlflow_run.info.artifact_uri
    config_artifact_uri = os.path.join(base_artifact_uri, callback.config_filename)
    assert yaml.safe_load(mlflow.artifacts.load_text(config_artifact_uri)) == namespace.as_dict()
