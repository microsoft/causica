# Zero-Shot Learning of Causal Models (Cond-FiP) 
[![Static Badge](https://img.shields.io/badge/paper-CondFiP-brightgreen?style=plastic&label=Paper&labelColor=yellow)
](https://arxiv.org/pdf/2410.06128)

This repo implements Cond-FiP proposed in the paper "Zero-Shot Learning of Causal Models". 

Cond-FiP is a transformer-based approach to infer Structural Causal Models (SCMs) in a zero-shot manner. Rather than learning a specific SCM for each dataset, we enable the Fixed-Point Approach (FiP) proposed in [Scetbon et al. (2024)](https://openreview.net/pdf?id=JpzIGzru5F), to infer the generative SCMs conditionally on their empirical representations. More specifically, we propose to amortize the learning
 of a conditional version of FiP to infer generative SCMs from observations and causal structures on synthetically generated datasets.

 Cond-FiP is composed of two models: (1) a dataset Encoder that produces embeddings given the empirical representations of SCMs, and (2) a Decoder that conditionnally on the dataset embedding infers the generative functional mechanisms of the associated SCM.

## Dependency
We use [Poetry](https://python-poetry.org/) to manage the project dependencies, they are specified in [pyproject](pyproject.toml) file. To install poetry, run:

```console
    curl -sSL https://install.python-poetry.org | python3 -
```
To install the environment, run `poetry install` in the directory of cond_fip project.


## Run experiments
In the [launchers](src/cond_fip/launchers) directory, we provide scripts to run the training of both the encoder and decoder.


### Amortized Learning of the Encoder
To train the Encoder on the synthetically generated datasets of [AVICI](https://arxiv.org/abs/2205.12934), run the following command:
```console
    python -m cond_fip.launchers.train_encoder
```
The model as well as the config file will be saved in `src/cond_fip/outputs`.


### Amortized Learning of Cond-FiP
To train the Decoder on the synthetically generated datasets of [AVICI](https://arxiv.org/abs/2205.12934), run the following command:
```console
    python -m cond_fip.launchers.train_cond_fip\
    --run_id <name_of_the_directory_containing_the_trained_encoder_model> 
```
The model as well as the config file will be saved in `src/cond_fip/outputs`. This command assumes that an Encoder model has been trained and saved in a directory located at `src/cond_fip/outputs/<name_of_the_directory_containing_the_trained_encoder_model>`.

### Test Cond-FiP on a new Dataset
To test a trained Cond-FiP, we also provide a [launcher file](src/cond_fip/launchers/inference_cond_fip.py), that enables to infer SCMs with Cond-FiP on new datasets.

To use this file, one needs to provide the path to the data in the [config file](src/cond_fip/config/numpy_tensor_data_module.yaml) by replacing the value of `data_dir`.
The data should respect a specific format. One can generate example of datasets by running:

```console
    python -m fip.data_generation.avici_data --func_type linear --graph_type er --noise_type gaussian --dist_case in --seed 1 --data_dim 5 --num_interventions 5
```
The data will be stored in `./data`.

To test a pre-trained Cond-FiP model on a specific dataset, one simply needs to run:
```console
    python -m cond_fip.launchers.inference_cond_fip\
    --run_id <name_of_the_directory_containing_the_pre_trained_model>\
    --path_data <path_to_the_data> 
```

This command assumes that a pre-trained Cond-FiP model has been saved in a directory located at `src/cond_fip/outputs/<name_of_the_directory_containing_the_pre_trained_model>`, and the data has been saved at the location `path_to_the_data`.


