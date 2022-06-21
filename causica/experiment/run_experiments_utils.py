import os
from typing import Any, Dict

from ..utils.io_utils import read_json_as


def build_model_config_dict(models_dir: str) -> Dict[str, Any]:
    # Build dictionary of model_config files for all the models
    model_config_dict = {}
    for f in os.scandir(models_dir):
        if not f.is_dir():
            continue
        model_dir = f.path
        config_path = os.path.join(model_dir, "model_config.json")
        if not os.path.exists(config_path):
            continue
        model_id = os.path.basename(model_dir)
        model_config = read_json_as(config_path, dict)
        model_config_dict[model_id] = model_config

    return model_config_dict
