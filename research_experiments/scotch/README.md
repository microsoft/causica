# Neural Structure Learning with Stochastic Differential Equations (SCOTCH) 
[![Static Badge](https://img.shields.io/badge/paper-SCOTCH-brightgreen?style=plastic&label=Paper&labelColor=violet)
](https://openreview.net/forum?id=V1GM9xDvIY)
[![Static Badge](https://img.shields.io/badge/Team-Causica-blue?style=plastic&label=Team&labelColor=purple)
]((https://www.microsoft.com/en-us/research/project/project_azua/people/))

This repo implements the SCOTCH proposed in the ICLR 2024 paper "Neural Structure Learning with Stochastic Differential Equations". 

SCOTCH is a structure learning method using neural stochastic differential equations (SDEs) for temporal data. SCOTCH, designed for the continuous-time processes, outperforms traditional discrete-time models, and is compatible with irregular sampling intervals. SCOTCH combines neural SDEs with variational inference over structures with theoretical guarantees regarding structure identifiability, establishing a new standard for structure learning. 

## Dependency
We use [Poetry](https://python-poetry.org/) to manage the project dependencies, they are specified in [pyproject](pyproject.toml) file. To install poetry, run:

```console
    curl -sSL https://install.python-poetry.org | python3 -
```
To install the environment, run `poetry install` in the directory of SCOTCH project.

## Prepare the data
To reproduce the experiment results in the [paper](https://openreview.net/forum?id=V1GM9xDvIY), you need to either generate the synthetic data (Lorenz and Yeast glycolysis dataset) or download the raw data and process them (DREAM3 and Netsim dataset). 
### Lorenz and Yeast data generation
To generate the synthetic data, run the following command:
```console
    python -m scotch.dataset_generation.generate_and_save_data
```
This executes the [generate_and_save_data.py](src/scotch/dataset_generation/generate_and_save_data.py) to generate Lorenz and Yeast datasets with 5 seeds, both normalized and unnormalized data, different sub-sampling rates and different missing data probabilities to mimic irregular sampling intervals. The generated data will be saved in the `./data/lorenz96_processed`.

### DREAM3
We use the DREAM3 dataset from [DREAM challenge](https://gnw.sourceforge.net/dreamchallenge.html#dream3challenge). One can download the zip file and extract the content to the `./data`. 

### Netsim
One can download the Netsim dataset from [Netsim](https://www.fmrib.ox.ac.uk/datasets/netsim/), and unzip the file to the `./data`. Then, specify the path to the `sim3.mat` in the file [generate_and_save_data.py](src/scotch/dataset_generation/generate_and_save_data.py#L59) and set `gen_netsim_data = True`. Run the following command to process the data:
```console
    python -m scotch.dataset_generation.generate_and_save_data
```
The processed data will be saved in `./data/netsim_processed`. If this already exists, it will override the data in `./data/netsim_processed`. 

## Run experiments
In the [src/scotch/experiments](src/scotch/experiments/) directory, we provide scripts to run the experiments for each datasets reported in the paper (Lorenz, Yeast, DREAM3 and Netsim). 

For `Ecoli1` dataset of DREAM3, run the following command:
```console
    python -m scotch.experiments.dream3  --dimension 100 --name Ecoli1 --epoch 40000 --lr 0.001 --sparsity 200 --dt 0.05 --seed 0 --normalize --experiment_name Ecoli1_exp --deci_diffusion --res_connection --sigmoid_output --lr_warmup 100
```
For other DREAM3 datasets, please refer to paper for hyperparameters. 

For `Netsim` dataset, run the following command:
```console
    python -m scotch.experiments.netsim --epoch 20000 --lr 0.001 --sparsity 1000 --dt 0.05 --seed 0 --res_connection --deci_diffusion --lr_warmup 100 --sigmoid_output --missing_prob 0.1 --experiment_name Netsim_missing_0.1
```

For `Lorenz` dataset, run the following command:
```console
    python -m scotch.experiments.lorenz --epoch 40000 --lr 0.003 --sparsity 500 --dt 1 --seed 0 --experiment_name Lorenz_missing_0.3 --res_connection --deci_diffusion --lr_warmup 100 --sigmoid_output --missing_prob 0.3 --num_time_points 100 --train_size 10
```

For `Yeast` dataset, run the following command:
```console
    python -m scotch.experiments.yeast --epoch 40000 --lr 0.001 --sparsity 200 --dt 1 --seed 0 --experiment_name Yeast_exp --res_connection --deci_diffusion --lr_warmup 100 --sigmoid_output --missing_prob 0.0 --num_time_points 100 --train_size 10 --normalize
```

