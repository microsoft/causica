import json

from causica.baselines.end2end_causal.end2end_causal import End2endCausal


def testsplit_configs(config_path):
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    if "model_hyperparams" in config:
        model_config = config["model_hyperparams"]
        # Some are weirdly of type list
        if isinstance(model_config, dict):
            # Check no errors
            End2endCausal.split_configs(model_config)
