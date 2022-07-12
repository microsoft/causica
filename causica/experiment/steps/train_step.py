from logging import Logger
from typing import Any, Dict, Union

from ...datasets.dataset import Dataset, SparseDataset
from ...datasets.variables import Variables
from ...models.imodel import IModel
from ...models.torch_model import TorchModel
from ...models_factory import create_model
from ..imetrics_logger import IMetricsLogger


def run_train_main(
    logger: Logger,
    model_type: str,
    output_dir: str,
    variables: Variables,
    metrics_logger: IMetricsLogger,
    dataset: Union[Dataset, SparseDataset],
    device: str,
    model_config: Dict[str, Any],
    train_hypers: Dict[str, Any],
) -> IModel:

    # Create model
    logger.info("Creating new model")
    model = create_model(model_type, output_dir, variables, device, model_config)
    if isinstance(model, TorchModel):
        num_trainable_parameters = sum(p.numel() for p in model.parameters())
        metrics_logger.set_tags({"num_trainable_parameters": num_trainable_parameters}, True)
    dataset.save_data_split(save_dir=model.save_dir)

    logger.info(f"Created model with ID {model.model_id}.")

    # Train model
    logger.info("Training model.")

    # TODO fix typing. mypy rightly complains that we may pass SparseDataset to a model that can only handle (dense) Dataset here.
    model.run_train(dataset=dataset, metrics_logger=metrics_logger, train_config_dict=train_hypers)  # type: ignore

    return model
