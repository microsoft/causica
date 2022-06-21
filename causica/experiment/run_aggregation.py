from typing import Any, Dict, List

from dependency_injector.wiring import Provide, inject

from ..experiment.steps.aggregation_step import run_aggregation_main
from .azua_context import AzuaContext


@inject
def run_aggregation(
    input_dirs: List[str],
    output_dir: str,
    experiment_name: str,
    aml_tags=Dict[str, Any],
    azua_context: AzuaContext = Provide[AzuaContext],
) -> None:
    _ = experiment_name
    metrics_logger = azua_context.metrics_logger()
    metrics_logger.set_tags(aml_tags)
    run_aggregation_main(input_dirs=input_dirs, output_dir=output_dir, metrics_logger=metrics_logger)

    metrics_logger.finalize()
