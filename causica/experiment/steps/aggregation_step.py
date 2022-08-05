"""
Aggregate results from multiple models.

See parallel_random_seeds_and_aggregate.py for example usage with AzureML.
"""

import copy
import glob
import logging
import os
from typing import Any, Dict, List

import mlflow
import pandas as pd

from ...datasets.variables import Variables
from ...utils.io_utils import flatten_keys, read_json_as, read_txt, save_json, unflatten_keys
from ...utils.run_utils import find_all_model_dirs

logger = logging.getLogger()


def run_aggregation_main(input_dirs: List[str], output_dir: str, summarise: bool = True) -> None:

    """
    Aggregate results from multiple directories corresponding to different model instances.

    Args:
        input_dirs: Directories to search for individual-model results
        output_dir : directory where the summary results should be saved.
        summarise: Flag indicating whether you want to create summary files with mean and standard deviation of each metric.
          If False, this function just creates a file all_results_and_configs.csv which has a column for each metric and
          config entry, and a row for each of the models.
          If True, you will also get summary JSON files and plots with mean and standard deviation of some metrics. Metrics
          are summarised in two ways: 1) over all runs, and 2) separately for each data split,
          in case you have run with multiple different random data splits.

    """
    os.makedirs(output_dir, exist_ok=True)
    model_dirs = find_all_model_dirs(input_dirs)
    if not model_dirs:
        logger.info("No models found")
        return

    results_and_configs = [flatten_keys(get_configs_and_results(d)) for d in model_dirs]
    df = pd.DataFrame(results_and_configs)

    # Write all results and configs to a CSV
    df.to_csv(
        os.path.join(output_dir, "all_results_and_configs.csv"),
        sep=",",
        encoding="utf-8",
    )

    if summarise:
        if len(set(x["dataset_name"] for x in results_and_configs)) > 1:
            # TODO summarise per dataset.
            # Aggregation of active learning metrics will break and means, stds will be meaningless.
            raise NotImplementedError

        _create_summary(df, output_dir, num_samples=len(model_dirs))


def _create_summary(
    df: pd.DataFrame,
    output_dir: str,
    num_samples: int,
):

    # Summary over all seeds
    summary_dict = _get_summary_dict_from_dataframe(df)
    result_fields = [
        "results",
        "target_results",
        "auic",
        "running_times",
        "system_metrics",
    ]
    _save_results(summary_dict, output_dir, result_fields)

    # Summary per data split
    summary_per_seed = _get_summary_dicts_from_groupby(df.groupby(by="dataset_config.random_seed"))
    for seed, seed_summary in summary_per_seed.items():
        seed_results_dir = os.path.join(output_dir, f"dataset_seed_{seed}")
        os.makedirs(seed_results_dir, exist_ok=True)
        _save_results(seed_summary, seed_results_dir, result_fields)

    def remove_value_by_key(summary, remove_key):
        if isinstance(summary, dict):
            summary.pop(remove_key, None)
            for _, val in summary.items():
                remove_value_by_key(val, remove_key)

    mlflow.log_metric("all_seeds.num_samples", num_samples)
    summary_dict_reduced = summary_dict.copy()
    remove_value_by_key(summary_dict_reduced, "num_samples")

    # Log metrics to AzureML
    if "results" in summary_dict_reduced:
        for x in ["train_data", "val_data", "test_data"]:
            if x in summary_dict_reduced["results"]:
                mlflow.log_metrics(
                    {f"all_seeds.{x}.Imputation MLL": summary_dict_reduced["results"][x].get("Imputation MLL", {})},
                )
                mlflow.log_metrics(
                    {f"all_seeds.{x}.all": summary_dict_reduced["results"][x].get("all", {})},
                )
    else:
        logger.info("No imputation results to log to AML")

    if "auic" in summary_dict_reduced:
        mlflow.log_metric("all_seeds.auic", summary_dict_reduced["auic"]["all"])
    else:
        logger.info("No AUIC to log to AML")
    if "running_times" in summary_dict:
        for x in summary_dict["running_times"]:
            mlflow.log_metrics(summary_dict["running_times"][x])
    else:
        logger.info("No running times to log to AML")
    if "system_metrics" in summary_dict:
        for x in summary_dict["system_metrics"]:
            mlflow.log_metrics(summary_dict["system_metrics"][x])
    else:
        logger.info("No system metrics to log to AML")


def _get_summary_dict_from_dataframe(df: pd.DataFrame) -> dict:
    """
    Given a dataframe with flattened keys as column headings, create a nested dict with
    {'mean': .., 'std': ..', 'num_samples': ...} at bottom level.
    """
    mean_df = df.mean()
    # Use biased std estimator, to match numpy default behaviour
    std_df = df.std(ddof=0)
    count_df = df.count()
    mean, std, count = [mean_df.to_dict(), std_df.to_dict(), count_df.to_dict()]
    return unflatten_keys({k: {"mean": mean[k], "std": std[k], "num_samples": count[k]} for k in mean})


def _get_summary_dicts_from_groupby(grouped_df) -> Dict[Any, Dict]:
    """
    Given a pandas GroupBy object, return a dict of summaries.
    Keys of the returned dict_of_summaries are the values the dataframe was grouped by,
    and the values are like the dicts you get from _get_summary_dict_from_dataframe.
    """
    mean_df = grouped_df.mean()
    # Use biased std estimator, to match numpy default behaviour
    std_df = grouped_df.std(ddof=0)
    count_df = grouped_df.count()
    mean, std, count = [mean_df.to_dict(), std_df.to_dict(), count_df.to_dict()]
    index = mean_df.index
    mean, std, count = [{i: {k: v[i] for k, v in d.items()} for i in index} for d in (mean, std, count)]
    dict_of_summaries = {}
    for i in index:
        keys = mean[i].keys()
        dict_of_summaries[i] = unflatten_keys(
            {k: {"mean": mean[i][k], "std": std[i][k], "num_samples": count[i][k]} for k in keys}
        )
    return dict_of_summaries


def _save_results(data: dict, save_dir: str, result_fields: List[str]):
    # Save results to JSON and save active learning plots.

    for field in result_fields:
        if field not in data:
            logger.info(f'No "{field}" found.  Perhaps imputation/AL did not run.')
            continue
        res = copy.deepcopy(data[field])
        save_json(unflatten_keys(res), os.path.join(save_dir, f"{field}.json"))


def get_configs_and_results(model_dir: str) -> dict:
    """
    Load configs and results from JSON files into a dict with keys 'model_config', 'train_config',
    'results', 'target_results' etc.

    """
    variables = Variables.create_from_json(os.path.join(model_dir, "variables.json"))
    configs_and_results: Dict[str, Any] = {}
    for k, pattern in {
        "model_config": "model_config.json",
        "train_config": "*/train_config.json",
        "dataset_config": "dataset_config.json",
        "results": "results.json",
        "target_results": "target_results.json",
        "auic": "active_learning/auic.json",
        "running_times": "running_times.json",
        "system_metrics": "system_metrics.json",
    }.items():
        fnames = glob.glob(os.path.join(model_dir, pattern))
        if len(fnames) > 1:
            raise ValueError
        elif fnames:
            configs_and_results.update({k: read_json_as(fnames[0], dict)})
        elif k in ["dataset_config", "model_config"]:
            raise FileNotFoundError
    configs_and_results.update({"model_type": read_txt(os.path.join(model_dir, "model_type.txt"))})
    configs_and_results.update({"dataset_name": read_txt(os.path.join(model_dir, "dataset_name.txt"))})
    configs_and_results["active_learning"] = {}
    for var in variables:
        if not var.query:
            filename = os.path.join(model_dir, "active_learning", f"{var.name}.json")
            if os.path.exists(filename):
                configs_and_results["active_learning"].update(read_json_as(filename, dict))

    return configs_and_results
