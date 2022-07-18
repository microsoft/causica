from typing import Any, Dict, List

from ..experiment.steps.aggregation_step import run_aggregation_main
from .run_context import RunContext


def run_aggregation(
    input_dirs: List[str],
    output_dir: str,
    experiment_name: str,
    aml_tags: Dict[str, Any],
    run_context: RunContext,
) -> None:
    _ = experiment_name
    metrics_logger = run_context.metrics_logger
    metrics_logger.set_tags(aml_tags)
    run_aggregation_main(input_dirs=input_dirs, output_dir=output_dir, metrics_logger=metrics_logger)

    metrics_logger.finalize()
