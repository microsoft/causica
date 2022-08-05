from typing import Any, Dict, List, Optional

import mlflow

from ..experiment.run_context import RunContext
from ..experiment.steps.aggregation_step import run_aggregation_main


def run_aggregation(
    input_dirs: List[str],
    output_dir: str,
    experiment_name: str,
    aml_tags: Dict[str, Any],
    run_context: Optional[RunContext] = None,
) -> None:
    # pylint: disable=unused-argument
    # run_context will be passed in by run_aml_step_from_kwargs_file
    mlflow.set_experiment(experiment_name)
    mlflow.set_tags(aml_tags)
    run_aggregation_main(input_dirs=input_dirs, output_dir=output_dir)
