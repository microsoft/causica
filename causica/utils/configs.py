import itertools
import os
from typing import Any, Dict, Optional, Tuple

from ..utils.io_utils import read_json_as, recursive_update


class ModelConfigNotFound(Exception):
    pass


def get_configs(
    model_type: str = "pvae",
    dataset_name: Optional[str] = None,
    override_model_path: Optional[str] = None,
    override_dataset_path: Optional[str] = None,
    override_impute_path: Optional[str] = None,
    override_objective_path: Optional[str] = None,
    default_configs_dir: str = "configs",
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load config files. For a given model, a set of 'global' default configs will first be loaded from
    [default_configs_dir]/defaults, and will then be updated with any values specified in dataset-specific config files
    found in [default_configs_dir]/dataset_name, if they exist. Finally, these configs will be updated with any values
    found in the user-specified configs loaded from the override_x_path arguments, if they are provided.

    Args:
        model_type: Name of model, e.g. 'pvae' or 'vaem'
        dataset_name: Name of dataset - will first look for a dataset + model config, then fall back to just model.
        override_model_path: Path to model config file with values to override.
        override_dataset_path: Path to dataset config file with values to override.
        override_impute_path: Path to imputation config file with values to override.
        override_objective_path: Path to objective config file with values to override.
        parameters_dir: Path to directory containing default parameters for models and datasets.

    Returns:
        Dictionaries containing:
        model_config, training_config, dataset_config, impute_config, objective_config
    """

    default_model_config_path = os.path.join(default_configs_dir, "defaults", f"model_config_{model_type}.json")
    if not os.path.exists(default_model_config_path):
        raise ModelConfigNotFound
    default_dataset_config_path = os.path.join(default_configs_dir, "defaults", "dataset_config.json")
    default_impute_config_path = os.path.join(default_configs_dir, "defaults", "impute_config.json")
    default_objective_config_path = os.path.join(default_configs_dir, "defaults", "objective_config.json")

    # Load global default configs for model.
    model_config = read_json_as(default_model_config_path, dict)
    dataset_config = read_json_as(default_dataset_config_path, dict)
    impute_config = read_json_as(default_impute_config_path, dict)
    objective_config = read_json_as(default_objective_config_path, dict)

    # Update with dataset-specific, configs if they exist
    if dataset_name is not None:
        dataset_config_dir = os.path.join(default_configs_dir, dataset_name)
        dataset_model_config_path = os.path.join(dataset_config_dir, f"model_config_{model_type}.json")
        dataset_dataset_config_path = os.path.join(dataset_config_dir, "dataset_config.json")
        dataset_impute_config_path = os.path.join(dataset_config_dir, "impute_config.json")
        dataset_objective_config_path = os.path.join(dataset_config_dir, "objective_config.json")

        model_config = update_dict_from_path(model_config, dataset_model_config_path)
        dataset_config = update_dict_from_path(dataset_config, dataset_dataset_config_path)
        impute_config = update_dict_from_path(impute_config, dataset_impute_config_path)
        objective_config = update_dict_from_path(objective_config, dataset_objective_config_path)

    # Update with user-specified configs, if they exist.
    model_config = update_dict_from_path(model_config, override_model_path)
    dataset_config = update_dict_from_path(dataset_config, override_dataset_path)
    impute_config = update_dict_from_path(impute_config, override_impute_path)
    objective_config = update_dict_from_path(objective_config, override_objective_path)

    return (
        model_config["model_hyperparams"],
        model_config["training_hyperparams"],
        dataset_config,
        impute_config,
        objective_config,
    )


def update_dict_from_path(old_dict, new_path):
    """
    Update the dictionary `old_dict` using a dictionary specified in a JSON file found at `new_path`, provided that the
    path `new_path` exists and is not None.

    Args:
        old_dict (dict): Dictionary to update
        new_path (pathlike or None): Path to JSON file containing values to update `old_dict` with. If None, no update
            will take place.

    Returns:
        old_dict (dict): The input dictionary, possibly updated.
    """
    if new_path is not None and os.path.exists(new_path):
        new_dict = read_json_as(new_path, dict)
        old_dict = recursive_update(old_dict, new_dict)
    return old_dict


def split_config(config, diagonal=False):
    """
    Splits a dictionary into a list of dictionaries. For each key that wants to be split
    the values should be provided in the form
                key : {"__split__" : True, values : list_of_values}
    and the returned list of dictionaries each contain key : val for each val in list_of_values.
    If multiple keys are to be split on, the returned list will produce an exhaustive
    product-list of dictionaries with each combination of values given in the list.
    If diagonal is set to True (default False), then each key to be split on must have value
    lists of the same length. The returned list of dicts will then be the same length as any
    given list_of_values and the i-th returned dictionary will contain the i-th value from each
    list_of_values.

    This method will split along keys within nested dictionaries too.

    e.g. (non-diagonal)
            {key_1 : {"__split__" : True, "values" : [val_1, val_2]},
             key_2 : [val_3, val_4],
             key_3 : {"__split__" : True, "values" : [val_5, val_6]}}
        becomes
            [{key_1 : val_1, key_2 : [val_3, val_4], key_3 : val_5},
             {key_1 : val_1, key_2 : [val_3, val_4], key_3 : val_6},
             {key_1 : val_2, key_2 : [val_3, val_4], key_3 : val_5},
             {key_1 : val_2, key_2 : [val_3, val_4], key_3 : val_6}]

    e.g. (diagonal)
            {key_1 : {"__split__" : True, values : [val_1, val_2]},
             key_2 : {"__split__" : True, values : [val_3, val_4]}}
        becomes
            [{key_1 : val_1, key_2 : val_3},
             {key_1 : val_2, key_2 : val_4}]

    Note: The value of dict["__split__"] is not actually checked so setting it to False
    will not prevent a split. To prevent a split, remove the "__split__" key entirely and
    pass the values as a list.
    Note: Order of returned dicts is not known.

    Args:
        config (dict): Dictionary to split.

    Returns:
        configs (list of dict): List of dictionaries.

    """

    split_dicts = []
    key_vals = []
    config_copy = config.copy()
    for key in config_copy:
        if isinstance(config_copy[key], dict):
            if "__split__" not in config_copy[key]:
                nested_dict_list = split_config(config[key], diagonal=diagonal)
                if len(nested_dict_list) == 1:
                    config_copy[key] = nested_dict_list[0]
                else:
                    config_copy[key] = {"__split__": True, "values": split_config(config[key], diagonal=diagonal)}
            if "__split__" in config_copy[key]:
                keyvals = []
                for val in config_copy[key]["values"]:
                    keyvals.append((key, val))
                key_vals.append(keyvals)

    if len(key_vals) == 0:
        return [config]

    if diagonal:
        key_val_iter = zip(*key_vals)
    else:
        key_val_iter = itertools.product(*key_vals)

    for key_val_list in key_val_iter:
        new_dict = config.copy()
        for key, val in key_val_list:
            new_dict[key] = val
        split_dicts.append(new_dict)

    return split_dicts
