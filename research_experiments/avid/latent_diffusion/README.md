# Adapting Video Diffusion Models to World Models (Latent Space)

Implements the latent diffusion experiments for RT1.

## Installing dependencies
We use [Poetry](https://python-poetry.org/) to manage the project dependencies, they are specified in [pyproject](pyproject.toml) file. To install poetry, run:

```console
    curl -sSL https://install.python-poetry.org | python3 -
```
To install the environment, run `poetry install` in avid/latent_diffusion.

Loading the RT1 data requires using tensorflow datasets. To avoid the dataloader consuming unnecessarily large amounts of CPU memory, we recommend using tcmalloc (see [here](https://github.com/tensorflow/tensorflow/issues/44176) for details). It can be installed using:
```
    sudo apt update
    sudo apt install libtcmalloc-minimal4
    export LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
```

## Downloading DynamiCrafter checkpoint
We use the 512 x 320 resolution version of [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter) as the pretrained base model. The checkpoint is available on HuggingFace and can be downloaded with:
```
    wget https://huggingface.co/Doubiiu/DynamiCrafter_512/resolve/main/model.ckpt
```
The AVID codebase expects the checkpoint to be located at /host_home/avid/dynamicrafter_512/model.ckpt. You will need to update the configs if the checkpoint is stored elsewhere.

## Training models
To train the models, run:
```
     ./scripts/train.sh --config configs/train/{CONFIG}.yaml --script scripts/train_{MODEL_TYPE}.py
```

For example, to train each of the 145M/170M models:
```
    ./scripts/train.sh --config configs/train/avid/avid_145M.yaml --script scripts/train_avid.py
    ./scripts/train.sh --config configs/train/control_net/control_net_lite_170M.yaml --script scripts/train_control_net.py 
    ./scripts/train.sh --config configs/train/act_cond_diffusion_145M.yaml --script scripts/train_diffusion.py 
```

The number of GPUs used for training is set in train.sh. Note that in our experiments we use 4 A100 GPUs to train each model, giving a global batch size of 64. Ensure to *adjust the batch size and gradient accumulation if you are using a different number of GPUs* for training.


## Evaluating models
Update the config files in config/eval to point to the correct checkpoint. To evaluate each of the baselines, the command is of the format:
```console
    python scripts/eval/eval_{MODEL_TYPE}.py --config config/eval/{CONFIG}.yaml
```

For example, to evaluate each of the 145M/170M models, you would run:
```console
    python scripts/eval/eval_avid.py --config configs/eval/avid_145M.yaml  
    python scripts/eval/eval_diffusion.py --config configs/eval/act_cond_diffusion_145M.yaml 
    python scripts/eval/eval_controlnet.py --config configs/eval/control_net_170M.yaml 
```