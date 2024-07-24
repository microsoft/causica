# A Fixed-Point Approach for Causal Generative Modeling (FiP) 
[![Static Badge](https://img.shields.io/badge/paper-FiP-brightgreen?style=plastic&label=Paper&labelColor=yellow)
](https://arxiv.org/pdf/2404.06969)

This repo implements FiP proposed in the ICML 2024 paper "A Fixed-Point Approach for Causal Generative Modeling". 

FiP is a transformer-based approach to learn Structural Causal Models (SCMs) from observational data. To do so, FiP uses an equivalent formulation of SCMs that does not require Directed Acyclic Graphs (DAGs), viewed as fixed-point problems on the causally ordered variables. To infer topological orders (TOs), we propose to amortize the learning of a TO inference method on synthetically generated datasets by sequentially predicting the leaves of graphs seen during training.

## Dependency
We use [Poetry](https://python-poetry.org/) to manage the project dependencies, they are specified in [pyproject](pyproject.toml) file. To install poetry, run:

```console
    curl -sSL https://install.python-poetry.org | python3 -
```
To install the environment, run `poetry install` in the directory of fip project.

## Prepare the data
To reproduce the results obtained in the [paper](https://arxiv.org/pdf/2404.06969), you need to generate the data. A more detailed explanation on how to generate the data can be found in [README.md](src/fip/data_generation/README.md).

### AVICI / Csuite / Causal NF data generation
To generate the [AVICI](https://arxiv.org/abs/2205.12934) synthetic data, run the following command:
```console
    bash src/fip/data_generation/avici_data.sh 
```
This executes the [avici_data.py](src/fip/dataset_generation/avici_data.py) file to generate various datasets from the dataset distributions presented in [AVICI](https://arxiv.org/abs/2205.12934). The generated data will be saved in the `src/fip/data`.

Similarly, to generate the [CSuite](https://arxiv.org/abs/2202.02195) and the [Causal NF](https://arxiv.org/abs/2306.05415) synthetic data, run the following commands:
```console
    bash src/fip/data_generation/csuite_data.sh 
    bash src/fip/data_generation/normalizing_data.sh 
``` 

## Run experiments
In the [launchers](src/fip/launchers) directory, we provide scripts to run the experiments reported in the paper. A more detailed explanation on how to use these files can be found in [README.md](src/fip/launchers/README.md).


### Zero-Shot Inference of TOs
To train the TO inference method on AVICI data, run the following command:
```console
    python -m fip.launchers.amortization
```
The model as well as the config file will be saved in `src/fip/outputs`.


### Learn FiP with (Partial) Causal Knowledge
To train FiP when the DAG is known, run the following command:
```console
    python -m fip.launchers.scm_learning_with_ground_truth 
    --ground_truth_case graph 
    --standardize 
```
The model as well as the config file will be saved in `src/fip/outputs`. If you want to train FiP, when only the TO is known, replace  `--ground_truth_case graph` with `--ground_truth_case perm`. These commands assume that the datasets have been generated and saved in `src/fip/data`.


### Learn FiP End-to-End
To train FiP end-to-end, run the following command:
```console
    python -m fip.launchers.scm_learning_with_predicted_truth
    --run_id <name_of_the_directory_containing_the_amortized_model>
    --standardize 
```
The model as well as the config file will be saved in `src/fip/outputs`. This command assumes that a TO inference model has been trained and saved in a directory located at `src/fip/outputs/<name_of_the_directory_containing_the_amortized_model>`. This command also assumes that the datasets have been generated and saved in `src/fip/data`.






















```