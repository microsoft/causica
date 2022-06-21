import os
from copy import deepcopy

from causica.utils.configs import get_configs, recursive_update, split_config, update_dict_from_path
from causica.utils.io_utils import save_json

file_dir = os.path.dirname(os.path.realpath(__file__))
tests_params_dir = os.path.join(file_dir, "parameters")


def test_get_default_config_pvae():
    configs = get_configs("pvae", default_configs_dir=tests_params_dir)
    model_config, train_config, dataset_config, impute_config, objective_config = configs
    assert "dep_embedding_dim" not in model_config
    assert model_config["embedding_dim"] == 10
    assert train_config["batch_size"] == 100
    assert dataset_config["use_predefined_dataset"] is False
    assert dataset_config["test_fraction"] == 0.1
    assert impute_config["sample_count"] == 50
    assert objective_config["sample_count"] == 50


def test_get_default_config_vaem():
    configs = get_configs("vaem", default_configs_dir=tests_params_dir)
    model_config, train_config, dataset_config, impute_config, objective_config = configs
    assert "embedding_dim" not in model_config
    assert model_config["dep_embedding_dim"] == 10
    assert train_config["batch_size"] == 100
    assert dataset_config["use_predefined_dataset"] is False
    assert dataset_config["test_fraction"] == 0.1
    assert impute_config["sample_count"] == 50
    assert objective_config["sample_count"] == 50


def test_get_default_config_pvae_specified_dataset():
    configs = get_configs("pvae", default_configs_dir=tests_params_dir, dataset_name="test_dataset")
    model_config, train_config, dataset_config, impute_config, objective_config = configs
    assert "dep_embedding_dim" not in model_config
    assert model_config["embedding_dim"] == 11
    assert train_config["batch_size"] == 101
    assert dataset_config["use_predefined_dataset"] is True
    assert impute_config["sample_count"] == 51
    assert objective_config["sample_count"] == 51


def test_get_default_config_vaem_specified_dataset():
    configs = get_configs("vaem", default_configs_dir=tests_params_dir, dataset_name="test_dataset")
    model_config, train_config, dataset_config, impute_config, objective_config = configs
    assert model_config["dep_embedding_dim"] == 11
    assert train_config["batch_size"] == 101
    assert dataset_config["use_predefined_dataset"] is True
    assert impute_config["sample_count"] == 51
    assert objective_config["sample_count"] == 51


def test_override_config():
    """
    Use default PVAE configs and incorporate changes from override configs.
    """
    configs = get_configs(
        "pvae",
        override_model_path=tests_params_dir + "/override_test/model_config.json",
        override_dataset_path=tests_params_dir + "/override_test/dataset_config.json",
        override_impute_path=tests_params_dir + "/override_test/impute_config.json",
        override_objective_path=tests_params_dir + "/override_test/objective_config.json",
        default_configs_dir=tests_params_dir,
    )
    model_config, train_config, dataset_config, impute_config, objective_config = configs
    assert "dep_embedding_dim" not in model_config
    assert model_config["embedding_dim"] == 10
    assert model_config["set_embedding_dim"] == 21
    assert train_config["batch_size"] == 100
    assert train_config["epochs"] == 1001
    assert dataset_config["use_predefined_dataset"] is False
    assert dataset_config["test_fraction"] == 0.3
    assert impute_config["sample_count"] == 52
    assert objective_config["sample_count"] == 52


def test_override_config_specified_dataset():
    """
    Use dataset + PVAE configs and incorporate changes from override configs.
    """
    configs = get_configs(
        "pvae",
        dataset_name="test_dataset",
        override_model_path=tests_params_dir + "/override_test/model_config.json",
        override_dataset_path=tests_params_dir + "/override_test/dataset_config.json",
        override_impute_path=tests_params_dir + "/override_test/impute_config.json",
        override_objective_path=tests_params_dir + "/override_test/objective_config.json",
        default_configs_dir=tests_params_dir,
    )
    model_config, train_config, dataset_config, impute_config, objective_config = configs
    assert "dep_embedding_dim" not in model_config
    assert model_config["embedding_dim"] == 11
    assert model_config["set_embedding_dim"] == 21
    assert train_config["batch_size"] == 101
    assert train_config["epochs"] == 1001
    assert dataset_config["use_predefined_dataset"] is False
    assert dataset_config["test_fraction"] == 0.3
    assert impute_config["sample_count"] == 52
    assert objective_config["sample_count"] == 52


def test_update_dict_from_path(tmpdir):
    old = {"a": {"a": 1, "b": 2}, "b": 2, "d": 4}
    new = {"a": {"a": 2, "c": 3}, "c": 3, "d": 5}
    expected = {"a": {"a": 2, "b": 2, "c": 3}, "b": 2, "c": 3, "d": 5}
    path = os.path.join(tmpdir, "config.json")
    save_json(new, path)
    updated = update_dict_from_path(old, path)
    assert updated == expected


def test_recursive_update():
    old_1 = {"a": {"a": 1, "b": 1}}
    old_2 = {}
    new_1 = {"a": {"a": 2, "c": 1}}
    new_2 = {"b": {"a": 1}}

    assert recursive_update(deepcopy(old_1), new_1) == {"a": {"a": 2, "b": 1, "c": 1}}
    assert recursive_update(deepcopy(old_1), new_2) == {"a": {"a": 1, "b": 1}, "b": {"a": 1}}
    assert recursive_update(deepcopy(old_2), new_1) == {"a": {"a": 2, "c": 1}}
    assert recursive_update(deepcopy(old_2), new_2) == {"b": {"a": 1}}


def test_split_config():
    config_to_split = {
        "val_1": 0,
        "list_to_split_1": {"__split__": True, "values": [0, 1]},
        "list_to_preserve": [10, 11],
        "nested_dict": {"val_1": 0, "val_2": 1},
        "nested_dict_with_list": {"list_1": {"__split__": True, "values": [3, 4]}},
    }
    split_configs = split_config(config_to_split)
    first_config = split_configs[0]
    assert len(split_configs) == 4
    assert first_config["list_to_preserve"] == [10, 11]
    assert first_config["list_to_split_1"] in [0, 1]  # It doesn't matter what order the configs are split in
    assert first_config["nested_dict_with_list"]["list_1"] in [3, 4]


def test_split_config_nested_list_only():
    config_to_split = {
        "val_1": 0,
        "nested_dict": {
            "nested_list": {"__split__": True, "values": [0, 1, 2]},
            "nested_list_2": {"__split__": True, "values": [3, 4]},
        },
        "nested_dict_2": {},
    }
    split_configs = split_config(config_to_split)
    first_config = split_configs[0]
    assert len(split_configs) == 6
    assert first_config["nested_dict"]["nested_list"] in [0, 1, 2]
    assert first_config["nested_dict"]["nested_list_2"] in [3, 4]
    assert first_config["nested_dict_2"] == {}


def test_split_config_diagonal():
    config_to_split = {
        "key_1": {"__split__": True, "values": [0, 5]},
        "nested_dict": {
            "nested_list": {"__split__": True, "values": [0, 1]},
            "nested_list_2": {"__split__": True, "values": [3, 4]},
        },
        "nested_dict_2": {},
    }
    split_configs = split_config(config_to_split, diagonal=True)
    assert len(split_configs) == 2
    assert split_configs[0]["key_1"] == 0
    assert split_configs[0]["nested_dict"]["nested_list"] == 0
    assert split_configs[0]["nested_dict"]["nested_list_2"] == 3
    assert split_configs[1]["key_1"] == 5
    assert split_configs[1]["nested_dict"]["nested_list"] == 1
    assert split_configs[1]["nested_dict"]["nested_list_2"] == 4
