# Adapting Video Diffusion Models to World Models (Pixel Space)

Implements the pixel-space diffusion experiments for procgen and coinrun.

## Installing dependencies
We use [Poetry](https://python-poetry.org/) to manage the project dependencies, they are specified in [pyproject](pyproject.toml) file. To install poetry, run:

```console
    curl -sSL https://install.python-poetry.org | python3 -
```
To install the environment, run `poetry install` in avid/pixel_diffusion.

## Generating datasets
To generate the procgen pretraining dataset and the coinrun datasets, in scripts/datasets run:
```console
    ./sample_procgen_dataset.sh
    ./sample_coinrun_datasets.sh
```

## Training models
To train the pretrained procgen model, run:
```console
    python scripts/train_diffusion.py --config config/base_video_config.yaml --config config/pretrained_model.yaml
```
For the results in the paper, we trained the pretrained model for 260k updates, taking 12 days on a single A100 GPU. Currently the pretrained model will be saved to and loaded from "/host_home/avid_checkpoints/pretrained_base_model/epoch=259-step=260000.ckpt". Update the configs as necessary if the model is stored elsewhere.

To train each of the baselines, the command is of the format:
```console
    python scripts/train_{MODEL_TYPE}.py --config config/{BASE_MODEL_TYPE_CONFIG}.yaml --config config/{MODEL_CONFIG}.yaml
```

For example, each of the 71M models can be trained using:
```console
    python scripts/train_avid.py --config config/avid/base_avid_config.yaml --config config/avid/avid_71M_coinrun_500lvl.yaml

    python scripts/train_control_net.py --config config/control_net/base_controlnet_config.yaml --config config/control_net/controlnet_full_coinrun_500lvl.yaml

    python scripts/diffusion.py --config config/base_video_config.yaml --config config/act_cond_model_71M_coinrun_500lvl.yaml
```


## Evaluating models
Update the config files in config/eval to point to the correct checkpoint. To evaluate each of the baselines, the command is of the format:
```console
    python scripts/eval/eval_{MODEL_TYPE}.py --config config/{BASE_MODEL_TYPE_CONFIG}.yaml --config config/{MODEL_CONFIG}.yaml --config/eval/{EVAL_CONFIG}.yaml 
```

For example:
```console
    python scripts/eval/eval_avid.py --config config/avid/base_avid_config.yaml --config config/avid/avid_71M_coinrun_500lvl.yaml --config config/eval/avid_71M_coinrun_500lvl.yaml
```