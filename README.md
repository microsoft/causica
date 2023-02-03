[![Causica CI Build](https://github.com/microsoft/causica/actions/workflows/ci-build.yml/badge.svg)](https://github.com/microsoft/causica/actions/workflows/ci-build.yml)

# Project Causica

## Overview
 
Causal machine learning enables individuals and organizations to make better data-driven decisions. In particular, causal ML allows us to answer “what if” questions about the effect of potential actions on outcomes. 
 
Causal ML is a nascent area, we aim  to enable a **scalable**, **flexible**, **real-world applicable end-to-end** causal inference framework. In perticular, we bridge between causal discovery, causal inference, and deep learning to achieve the goal.  We aim to develop technology can automate causal decision-making using existing observational data alone, output both the discovered causal relationships and estimate the effect of actions simultaneously.
 
Causica is a deep learning library for end-to-end causal inference, including both causal discovery and inference.  It implements deep end-to-end inference framework [2] and different alternatives.
 
This project splits the interventional decision making from observational decision making Azua repo found here [Azua](https://github.com/microsoft/project-azua).

[1] Alvarez et al. [Simultaneous Missing Value Imputation and Structure Learning with Groups](https://openreview.net/pdf?id=4rm6tzBjChe), NeurIPS 2022

[2] Geffner et al. [Deep End-to-end Causal Inference.](https://arxiv.org/pdf/2202.02195.pdf)

[3] Gong et al.  [Rhino: Deep Causal Temporal Relationship Learning With History-dependent Noise](https://openreview.net/pdf?id=i_1rbq8yFWC), ICLR 2023

[4] Ma et al. [Causal Reasoning in the Presence of Latent Confounders via Neural ADMG Learning]( https://openreview.net/pdf?id=dcN0CaXQhT), ICLR 2023

# DECI: End to End Causal Inference

## About

Real-world data-driven decision making requires causal inference to ensure the validity of drawn conclusions. However, it is very uncommon to have a-priori perfect knowledge of the causal relationships underlying relevant variables. DECI allows the end user to perform causal inference without having complete knowledge of the causal graph. This is done by combining the causal discovery and causal inference steps in a single model. DECI takes in observational data and outputs ATE and CATE estimates. 

For more information, please refer to the [paper](https://arxiv.org/abs/2202.02195).


**Model Description**

DECI is a generative model that employs an additive noise structural equation model (ANM-SEM) to capture the functional relationships among variables and exogenous noise, while simultaneously learning a variational distribution over causal graphs. Specifically, the relationships among variables are captured with flexible neural networks while the exogenous noise is modelled as either a Gaussian or spline-flow noise model. The SEM is reversible, meaning that we can generate an observation vector from an exogenous noise vector through forward simulation and given a observation vector we can recover a unique corresponding exogenous noise vector. In this sense, the DECI SEM can be seen as a flow from exogenous noise to observations. We employ a mean-field approximate posterior distribution over graphs, which is learnt together with the functional relationships among variables by optimising an evidence lower bound (ELBO). Additionally, DECI supports learning under partially observed data.

**Simulation-based Causal Inference**

DECI estimates causal quantities (ATE / CATE) by applying the relevant interventions to its learnt causal graph (i.e. mutilating incoming edges to intervened variables) and then sampling from the generative model. This process involves first sampling a vector of exogenous noise from the learnt noise distribution and then forward simulating the SEM until an observation vector is obtained. ATE can be computed via estimating an expectation over the effect variable of interest using MonteCarlo samples of the intervened distribution of observations. DECI can not simulate samples from conditional intervened distributions directly. However, we can learn an estimator of the conditional intervened distribution from samples of the joint distribution of conditioning variables and effect variables. Specifically, we use radial-basis-function kernel regression to evaluate expectations in the CATE formula.


**Specifying a prior over graphs**

If some a-prior knowledge of the causal graph is available, it can be incorporated as a prior for DECI by using the `informed_deci` model. The a-prior graph should is passed to DECI as a part of the `CausalDataset`. The hyperparameters used to specify strength of belief in the prior graph are discussed below. 


**Specifying a noise model**

If the exogenous noise is known to be Gaussian, we recommend employing the `deci_gaussian` model. This model has its Gaussian exogenous noise distribution mean set to 0 while its variance is learnt.

In the more general setting, we recommend using `deci_spline`. Here the exogenous noise distribution is a flexible spline flow that is learnt from the data. This model provides most gains in heavy-tailed noise settings, where the Gaussian model is at risk of overfitting to outliers. 

## How to run

The command to train this model with the csuite_symprod_simpson dataset and evaluate causal discovery and (C)ATE estimation performance is:

```
python -m causica.run_experiment csuite_symprod_simpson --model_type deci_spline --model_config configs/deci/true_graph_deci_spline.json -dc configs/dataset_config_causal_dataset.json -c -te
```

## How to generate data for experiments

We use different data for each of our 3 experiments: large scale evaluation data, csuite synthetic data and Twins/IHDP

**Large scale evaluation**

There's three different types of data we use:
1. Synthetic. Samples a random graph with N nodes and E edges and generates the data using gaussian processes for the functions in the SEM. Go to `causica/data_generation/large_synthetic` and run `python data_generation.py`. This will store the datasets generated in the `causica/data_generation/large_synthetic/data/` directory.
2. Pseudo-real. Syntren, a simulator that creates synthetic transcriptional regulatory networks and produces simulated gene expression data that approximates experimental data. To generate the data open the file `causica/data_generation/syntren/generate.py` and follow the instructions within.
3. Real. Protein cells datasets, from Sachs et. al. 2005. The data measures the expression level of different proteins and phospholipids in human cells. To generate the data open the file `causica/data_generation/protein_cells/generate.py` and follow the instructions within.

**CSuite**

The CSuite datasets can generated by running `python -m causica.data_generation.csuite.simulate` from the top-level directory of the package.
The datasets are created inside `data/`.



**Twins & IHDP**

We also include two causal inference benchmark datasets for ATE evaluation: the `twins` (twin birth datasets in the US) 
dataset and `IHDP` dataset (Health and Development Program data). Both are semi-synthetic datasets that are processed based 
on similar procedures as described in (Louizos et al., 2017), as detailed below.

- `IHDP` dataset. This can be generated by simply calling `research_experiments/load_ihdp_to_generate_CATE_dataset.py
`. This dataset contains measurements of both infants and their mother during real-life data collected in a randomized experiment. 
The main task is to estimate the effect of home visits by specialists on future cognitive test scores of infants. 
The outcomes of treatments are simulated artificially as in (Hill et al., 2011) hence both the factual and counterfactual outcomes 
of both treatments (home visits or not) on each subject are known (counterfactuals for evaluation only).
To make the task more challenging, additional confoundings are manually introduced by removing a subset (non-white mothers) of the treated children population. 
In this way we can construct the `IHDP` dataset of 747 individuals with 6 continuous covariates and 19 binary covariates. All continuous covariates are normalized. 

- `Twins` dataset. This can be generated by calling `research_experiments/load_twins_to_generate_CATE_dataset.py
`. This dataset consists of twin births in the US between 1989-1991. Only twins which with the same sex born weighing less 
than 2kg are considered. The treatment is defined as being born as the heavier one in each twins pair, and the outcome is 
defined as the mortality of each twins in their first year of life. Therefore, by definition for each pair of twins, we 
both factual and counterfactual outcomes of the treatments (counterfactuals for evaluation only). Following (Louizos et al, 2017), 
we also introduce artificial confounding using the categorical `GESTAT10` variable. All continuous covariates are normalized.  

## Reproducing experimental results

To reproduce all results (with the pipeline) you will need to generate all the data, as described above, and run all simulations as described here.

Methods are specified using a combination of the `--model_type` and `-model_config` arguments to the `run_experiment.py` script.
See the section for [IHDP and Twins Benchmarks](#ihdp-and-twins-benchmarks) for illustrative examples of the commands that are used.

Once runs are finished, all experimental results can be downloaded by pointing the script `research_experiments/e2e_causal_inference/notebooks/download_results.py` at the experiment name of interest on Azure ML. This will download the results to a local folder of choice. These results can then be loaded using functionality in `research_experiments/e2e_causal_inference/notebooks/utils.py`. We provide a number of notebooks which load this data and generate the plots in the paper "Deep End to End Causal Inference" as described below.

### IHDP and Twins Benchmarks

To run DECI on `IHDP` and `twins` datasets, simply run one of the following, using IHDP ATE as an example (for CATE estimation, 
or twins dataset, simply change `ihdp_norm` to `ihdp_norm_cate`, `twins`, or `twins_cate`):

* DECI Gaussian on IHDP for ATE estimation:

```
python -m causica.run_experiment ihdp_norm --model_type deci_gaussian --model_config configs/deci/deci_gaussian.json -te -dc configs/dataset_config_causal_dataset.json 
```

* DECI Spline on IHDP for ATE estimation:

```
python -m causica.run_experiment ihdp_norm --model_type deci_spline --model_config configs/deci/deci_spline.json -te -dc configs/dataset_config_causal_dataset.json 
```

* True graph DECI Gassian on IHDP for ATE estimation:

```
python -m causica.run_experiment ihdp_norm --model_type deci_gaussian --model_config configs/deci/true_graph_deci_gaussian.json -te -dc configs/dataset_config_causal_dataset.json 
```

* True graph DECI Spline on IHDP for ATE estimation:

```
python -m causica.run_experiment ihdp_norm --model_type deci_spline --model_config configs/deci/true_graph_deci_spline.json -te -dc configs/dataset_config_causal_dataset.json 
```

* DECI Gaussian DoWhy linear on IHDP for ATE estimation:

```
python -m causica.run_experiment ihdp_norm --model_type deci_dowhy --model_config configs/deci/deci_gaussian_dowhy_linear.json -te -dc configs/dataset_config_causal_dataset.json 
```

* DECI Spline DoWhy linear on IHDP for ATE estimation:

```
python -m causica.run_experiment ihdp_norm --model_type deci_dowhy --model_config configs/deci/deci_spline_dowhy_linear.json -te -dc configs/dataset_config_causal_dataset.json
```

* True graph DoWhy linear on IHDP for ATE estimation:

```
python -m causica.run_experiment ihdp_norm --model_type true_graph_dowhy --model_config configs/deci/dowhy_linear.json -te -dc configs/dataset_config_causal_dataset.json
```


* DECI Gaussian DoWhy nonlinear on IHDP for ATE estimation:

```
python -m causica.run_experiment ihdp_norm --model_type deci_dowhy --model_config configs/deci/deci_gaussian_dowhy_nonlinear.json -te -dc configs/dataset_config_causal_dataset.json
```

* DECI Spline DoWhy nonlinear on IHDP for ATE estimation:

```
python -m causica.run_experiment ihdp_norm --model_type deci_dowhy --model_config configs/deci/deci_spline_dowhy_nonlinear.json -te -dc configs/dataset_config_causal_dataset.json
```

* True graph DoWhy nonlinear on IHDP for ATE estimation:

```
python -m causica.run_experiment ihdp_norm --model_type true_graph_dowhy --model_config configs/deci/dowhy_nonlinear.json -te -dc configs/dataset_config_causal_dataset.json
```

* PC DoWhy linear on IHDP for ATE estimation:

```
python -m causica.run_experiment ihdp_norm --model_type pc_dowhy --model_config configs/deci/pc_dowhy_linear.json -te -dc configs/dataset_config_causal_dataset.json
```


* PC DoWhy nonlinear on IHDP for ATE estimation:

```
python -m causica.run_experiment ihdp_norm --model_type pc_dowhy --model_config configs/deci/pc_dowhy_nonlinear.json -te -dc configs/dataset_config_causal_dataset.json
```

### CSuite

To launch continuous CSuite experiments, use a command of the form

```
python -m causica.run_experiment <dataset> --model_type <model_type> --model_config <model_config> -c -te -dc configs/dataset_config_causal_dataset.json 
```
where `<dataset>` should be one of
 - `csuite_lingauss`,
 - `csuite_linexp`,
 - `csuite_nonlingauss`,
 - `csuite_nonlin_simpson`,
 - `csuite_symprod_simpson`,
 - `csuite_large_backdoor`,
 - `csuite_weak_arrows`.
The `--model_type` and `--model_config` arguments can be specified as in the section for [IHDP and Twins Benchmarks](#ihdp-and-twins-benchmarks), but
note the use of the `-c` flag for CSuite datasets to evaluate causal discovery.

To launch discrete CSuite experiments, the `-el` argument should also be passed, giving
```
python run_experiment.py <dataset> --model_type <model_type> --model_config <model_config> -c -te -el -dc configs/dataset_config_causal_dataset.json 
```
where `<dataset>` should be one of
 - `csuite_cat_to_cts`,
 - `csuite_cts_to_cat`,
 - `csuite_cat_collider`,
 - `csuite_cat_chain`,
 - `csuite_mixed_simpson`,
 - `csuite_large_backdoor_binary_t`,
 - `csuite_weak_arrows_binary_t`,
 - `csuite_mixed_confounding`.
The `--model_type` and `--model_config` arguments can again be specified as in the section for [IHDP and Twins Benchmarks](#ihdp-and-twins-benchmarks).

The illustrative plots from the paper can be generated using `research_experiments/e2e_causal_inference/notebooks/CATE_plots.ipynb`. However, this requires DECI and true_graph_DECI to have been run locally or the complete run output folders to have been downloaded from Azure ML.

### Systematic evaluation on large-scale graphs

To launch large-scale graph experiments, use a command of the form

```
python -m causica.run_experiment <dataset> --model_type <model_type> --model_config <model_config> -c <-te> -dc configs/dataset_config_causal_dataset.json 
```
where `<dataset>` should either be `{dataset}_seed_{seed}_std`, with
- `{dataset}`: either `protein_cells` or `syntren`
- `{seed}`: the seed used for dataset splitting, in `[1, 5]`
- A `miss_` prefix can be appended for experiments with partial observability

or follow this convention for ER and SF graphs `{graph_type}_{N}_{E}_spline_sem_{noise_type}_noise_{seed}_seed`, with:
- `{graph_type}`: either `ER` or `SF`
- `{N}`: the number of nodes; and `{E}`: the number of edges - we considered `{N}` in `[16, 64, 128]` and `{E}` in `[N, 4N]`
- `{noise_type}`: either `fixed` (for Gaussian noise) or `mlp` (for non-Gaussian noise)
- `{seed}`: the seed used for the dataset generation - we used the range `[0, 4]`
- An additional `_partial` postfix for experiments with partial observability

The `--model_type` and `--model_config` arguments can be specified as in the section for [IHDP and Twins Benchmarks](#ihdp-and-twins-benchmarks), but
note the use of the `-c` and `-te` flags for causal discovery evaluation and treatment effect evaluation, respectively. Note, the `-te` flat is not used for the `protein` or `syntron` datasets.

A full example would be:
```
python -m causica.run_experiment SF_64_64_spline_sem_mlp_noise_4_seed --model_type deci_dowhy --model_config configs/deci/deci_spline_dowhy_nonlinear.json -c -te -dc configs/dataset_config_causal_dataset.json 
```
### Prior graph experiments

Experiments are launched by calling `python evaluation_pipeline/runs/run_icml_inform.py`

Once results are downloaded locally, they can be viewed with `research_experiments/e2e_causal_inference/notebooks/prior_plots.ipynb`. Due to a slightly different experiment structure, this notebook uses helper functions stored in `research_experiments/e2e_causal_inference/notebooks/prior_notebook_utils.py`.

## Hyperparameters

Model configs:

- `lambda_dag`: Coefficient for the prior term that enforces the learnt graph to be a DAG.
- `lambda_sparse`: Coefficient for the prior term that enforces similarity to the prior matrix W_0 or sparsity in the case of an empty prior matrix.
- `tau_gumbel`: Temperature for the gumbel softmax trick.
- `spline_bins`: How many bins to use for spline flow base distribution if the 'spline' choice is made.
- `layers_imputer`: Number and size of hidden layers for imputer NN (e.g. [20,10,20] for a 3 hidden layer MLP with 20,10 and 20 neurons per hidden layer).
- `var_dist_A_mode`: Variational distribution for adjacency matrix. Admits {"simple", "enco", "true"}. "simple"
                             parameterizes each edge (including orientation) separately. "enco" parameterizes
                             existence of an edge and orientation separately. "true" uses the true graph.
- `norm_layers`: bool indicating whether all MLPs should use layer norm
- `res_connection`:  bool indicating whether all MLPs should use layer norm
- `layers_encoder`: Optional list indicating width of layers in GNN encoder MLP. If unspecified, defaults to a size dependent on size of observation vector.
- `layers_decoder`: Optional list indicating width of layers in GNN decoder MLP.  If unspecified, defaults to a size dependent on size of observation vector.
- `cate_rff_n_features`: number of random features to use in function approximation when estimating CATE. 
- `cate_rff_lengthscale`: lengthscale of RBF kernel used when estimating CATE.
- `prior_A`: prior adjacency matrix. Does not need to be a DAG.
- `prior_A_confidence`: degree of confidence in prior adjacency matrix enabled edges between 0 and 1. e.g. a value of 0.5 enforces a belief that the entries in prior_A are correct 50% of the time.

Training hyperparameters:

- `learning_rate`: Learning rate used at the beginning of each augmented Lag step
- `likelihoods_learning_rate`: Learning rate used for noise models
- `batch_size`: Batch size used
- `stardardize_data_mean`: Whether to center data or not
- `stardardize_data_std`: Whether to standardize data to have unit variance or not
- `rho`: Initial rho for the auglag procedure. This is the value multiplying the square of the dag penalty
- `alpha`: Initial alpha for the auglag procedure. This is the term multiplying the linear dag penalty
- `safety_rho`: Max value of rho allowed
- `safety_alpha`: Max value of alpha allowed
- `max_steps_auglag`: Max number of auglag steps allowed
- `max_p_train_dropout`: Max probability for artificially dropping data during training to regularize imputation network
- `anneal_entropy`: Admits "linear" and "noanneal". With the latter the entropy term in the ELBO (of q(A)) is left without any changes, with the former it is annealed during training. More specifically, for the "auglag" optimization mode, it annealed with a factor 1/step after the 5th auglag step.
- `progress_rate`: minimum ratio of relative decrease in DAG penalty from one outer loop step to the next of augmented Lag optimisation. If the progress_rate is not met, then the strength of the quadratic DAG penalty is increased.
- `max_auglag_inner_epochs`: Number of optimisation steps to take in inner loop of augmented Lag optimisation
- `max_epochs_per_step`: Max number of epochs allowed per auglag step
- `reconstruction_loss_factor`: Multiplier for reconstruction loss in partially observed data setting

## Further extensions 

### Deconfounded DECI (D-DECI, also referred as N-ADMG in our paper)

This method is based on our recent accepted ICLR 2023 paper (Causal Reasoning in the Presence of Latent Confounders 
via Neural ADMG Learning, ICLR 2023), which is an extension of DECI to relax the assumption of causal 
sufficiency. In this model, we assume all latent variables are assumed to be confounders, and inference is performed 
over them using amortized variational inference. See [our paper](https://openreview.net/forum?id=dcN0CaXQhT) for 
details. 

In this repo, we implement four versions of D-DECI/N-ADMG:
- `ADMGParameterisedDDECIGaussian` uses an ADMG parameterisation over observed variables, which consists of both directed 
  edges (indicating direct causal associations) and bidirected edges (indicating that a latent confounder acts 
  between a pair of observed variables). Meanwhile, we assume gaussian additive noise for the model.
- `ADMGParameterisedDDECISpline` The spline additive noise counterpart of `ADMGParameterisedDDECIGaussian`.
- `BowFreeDDECIGaussian` also uses an ADMG parameterisation, but further assumes that each latent variable is a 
  parent-less confounder of a pair of non-adjacent observed variables. This further implies that the underlying ADMG 
  must be bow-free (both a directed and a bidirected edge cannot exist between the same pair of observed variables). 
  This is an assumption that is necessary for structural identifiability.  
- `BowFreeDDECISpline` the spline additive noise counterpart of `BowFreeDDECIGaussian`.

All four versions train model parameters while performing approximate inference over the adjacency matrices ($q_
{\phi}(G)$) and the latent varaibles (through an inference network $q_{\phi}(u | x)$).  

#### Example usage

Generally the D-DECI code works in a similar way to DECI and has mainly the same api. To run the model and reproduce
the result of the synthetic data in the paper (experiment 1), simply run:

```python
python -m causica.run_experiment.py csuite_fork_collider_nonlin_gauss_latent_confounder --model_type 
bowfree_ddeci_gaussian --model_config configs/ddeci/bowfree_ddeci_gaussian.json -dc 
configs/dataset_config_latent_confounded_causal_dataset.json -lcc  -te 
```
where `csuite_fork_collider_nonlin_gauss_latent_confounder` is the name for the synthetic fork-collider dataset 
designed for causal discovery and inference under latent confounding, and `bowfree_ddeci_gaussian` is the model type 
name for `BowFreeDDECIGaussian`. `-lcc` is the flag for evaluating causal discovery under latent confounders.


#### Running on your on data
Any dataset of your own that is prepared in the format of DECI can be runned by D-DECI, by simply replacing the 
dataset name in the example usage command.dataset name in the examplar usage command.

#### Additional model config hyper parameters:

D-DECI shares most of the model config hyper parameters with DECI, with the following exceptions:
- `anneal_beta`: Admits "linear" and "reverse". "linear" applies linear annealing to the KL term between the 
  approximate posterior  and prior over the latent variables, updating between auglag steps.; and "reverse" will 
  increase the beta values linearly instead of decrease.
- `anneal_beta_max_steps`: The number of steps we apply KL annealing for.
- `beta`: Constant by which the KL term is multiplied.


# Rhino: Deep Causal Temporal Relationship Learning with History-dependent Noise

## About
Discovering causal relationships between different variables from time series data
has been a long-standing challenge. We propose Rhino, an extension to DECI, that is capable of performing both temporal causal discovery and inference. 
It allows the user to provide pure observational time-series data and specification of treatment-effect variable pair with conditioning history, and Rhino can infer the underlying
temporal causal graphs with the target treatment effect values (i.e. CATE). The key features of Rhino are
1. It can model nonlinear relationships between variables.
2. It can model both lagged and instantaneous effects.
3. More importantly, it allows the noise distribution to be history-dependent. Namely, the observation noise distribution can change according to the past actions.
4. Rhino is theoretically sound, namely, it is structurally identifiable. 

For more information, please refer to [Rhino paper](https://openreview.net/forum?id=i_1rbq8yFWC).

### Capability
It can perform both temporal causal discovery and inference. One important underlying assumption of Rhino is that the underlying temporal causal structure
is causal stationary, namely, the causal connection does not change w.r.t. the absolute time. 

After fitting Rhino to the observational data, it can also perform the simulation-based causal inference. Since Rhino is an extension of DECI, which is also based on SEM,
ancestral sampling can be used to draw samples from the intervention distribution. 

### Noise model and instantaneous effect
Currently, Rhino supports both enabling/disabling instantaneous effect and history-dependent noise. Specifically, these can be set via the model config files.
`allow_instantaneous` (`True` for enabling instantaneous effect) and `base_distribution_type`, which controls the noise distribution type. 
Currently, Rhino supports `gaussian`, `spline`, and `conditional_spline`, where the former two disable the history-dependent noise. If `conditional_spline` is chosen, it can also be further configured. See [Hyperparameter](##Hyperparameter). 
In the current version of Rhino, the causal inference with `conditional_spline` noise type performs slightly worse than `gaussian` and `spline` noise. 

### Adjacency matrix
Rhino also supports the temporal causal graphs. The temporal causal graph are represented using temporal adjacency matrix. It is a binary tensor, called `adj_matrix`,  with shape `[lag+1, num_nodes, num_nodes]`,
where `lag` is the user specified model lag, and `num_nodes` is the dimensionality. `adj_matrix[0]` represents the causal connection in the current time step, and `adj_matrix[i]` for `i>0`, represents the causal connection from `t-i` to current `t`. 
For example, `adj_matrix[0,l,k] = 1` represents `x_{t,l} -> x_{t,k}` and `adj_matrix[1,l,k]=1` represents `x_{t-1,l} -> x_{t,k}`.

Ground truth adjacency matrix are typically stored as `adj_matrix.npy` in the dataset folder. 

## Dataset
In the [paper](https://openreview.net/forum?id=i_1rbq8yFWC), we use three types of dataset: `Synthetic`, `DREAM3` and `Netsim`. We only provide `Synthetic` dataset generation at `causica/data_generation/temporal_synthetic`. For the other two dataset, one can download 
the raw DREAM3 data [here](http://gnw.sourceforge.net/resources/DREAM3%20in%20silico%20challenge.zip) and Netsim [here](https://github.com/sakhanna/SRU_for_GCI/tree/master/data/netsim). The preprocessing can follow [Use you own data](###Use you own data).

### Generate synthetic dataset
To generate the synthetic dataset, one can run 
```python
python -m causica.data_generation.temporal_synthetic.generate_all_cts
```
The generated datasets will be stored at `./data` folder. 

### Use your own data
Rhino can handle the time-series data. The time-series data should be stored in a `.csv` file called `train.csv`, where each entry should be separated by delimiter `,`. The `csv` file should not have the header.
The first column in the csv file must be the time-series index. The other columns can indicate the variables. Within the same time-series index, the row should be sorted by the ascending order of time. 
For example, if we have two one-dimensional series with data `s0= [1,2,3,4]` and `s1 = [9,8,7]` sorted by time. Then, the csv should look like

| 0 | 1 |
|---|---|
| 0 | 2 |
| 0 | 3 |
| 0 | 4 |
| 1 | 9 |
| 1 | 8 |
| 1 | 7 |

This file should be stored in a folder at `.\data` in repo root directory. 

## How to run
To run Rhino with dataset `ER_ER_lag_2_dim_10_HistDep_0.5_mlp_spline_product_con_1_inthist_3_seed_0`, one can run the following
```python
python -m causica.run_experiment ER_ER_lag_2_dim_10_HistDep_0.5_mlp_spline_product_con_1_inthist_3_seed_0 --model_type rhino -dc configs/dataset_config_temporal_causal_dataset.json --model_config configs/rhino/model_config_rhino_synthetic.json -dv gpu -c
```
where `--model_type` specifies the model name, `-dc` specifies the dataset config and `--model_config` specifies the location of model config file. `-dv` specifies the device to use, can choose `cpu` or `gpu`, `-c` specifies that we want to evaluate the causal discovery performance, provided that we have the ground truth adjacency matrix.

### Reproducing the results
To reproducing the results in the paper, especially for DREAM3 and Netsim, after preprocessing the data into the correct format and store them in `./data` folder. Then,
we include the configs files in `./configs/rhino/`. Just need to use the correct config file in `--model_config`.

## Hyperparameter
To tune the model, one needs to change the model config files. They are similar to the one for DECI, but we have additional hyperparameter. 

- `ICGNN_embedding_size`: The embedding size for the ICGNN, see DECI formulation, use `null` for default choice.
- `additional_spline_flow`: The number of additional diagonal afine flow used after the conditional spline flow to increase flexibility.
- `allow_instantaneous`: Whether the model incorporates instantaneous effect or not.
- `base_distribution_type`: The noise distribution type.
- `conditional_decoder_layer_sizes`: The layer size of the decoder in ICGNN for conditional spline flow. 
- `conditional_embedding_size`: The embedding size of ICGNN for conditional spline flow.
- `conditional_encoder_layer_sizes`: The layer size of the encoder in ICGNN for conditional spline flow.
- `conditional_spline_order`: The order of the conditional spline flow, `linear` or `quadratic`.
- `disable_diagonal_eval`: whether we ignore the self connection in the temporal adj matrix during evaluation. This is only applied to the evaluation of aggregated adj matrix. E.g. DREAM3 or Netsim evaluation.
- `init_logits`: The initialized logits of the graph posterior variational distribution. Negative values prefer dense graph. 
- `lag`: Model lag hyperparameter.
- `var_dist_A_mode`: The variational distribution used for graph learning. Currently, for time series, it only supports `temporal_three`.

The model training hyperparameter are defined exactly the same as DECI.


# References

If you have used the models in our code base, please consider to cite the corresponding papers:

[1], **(VISL)** Pablo Morales-Alvarez, Wenbo Gong, Angus Lamb, Simon Woodhead, Simon Peyton Jones, Nick Pawlowski, Miltiadis Allamanis, Cheng Zhang, "Simultaneous Missing Value Imputation and Structure Learning with Groups", [ArXiv preprint](https://arxiv.org/abs/2110.08223)

[2], **(DECI)** Tomas Geffner, Javier Antoran, Adam Foster, Wenbo Gong, Chao Ma, Emre Kiciman, Amit Sharma, Angus Lamb, Martin Kukla, Nick Pawlowski, Miltiadis Allamanis, Cheng Zhang. Deep End-to-end Causal Inference. [Arxiv preprint](https://arxiv.org/abs/2202.02195) (2022)

[3], **(DDECI)** Matthew Ashman, Chao Ma, Agrin Hilmkil, Joel Jennings, Cheng Zhang. Causal Reasoning in the Presence of Latent Confounders via Neural ADMG Learning. [ICLR](https://openreview.net/forum?id=dcN0CaXQhT) (2023)

[4], **(Rhino)** Wenbo Gong, Joel Jennings, Cheng Zhang, Nick Pawlowski. Rhino: Deep Causal Temporal Relationship Learning with History-dependent Noise. [ICLR](https://openreview.net/forum?id=i_1rbq8yFWC) (2023)

Other references:
- Louizos, Christos, et al. "Causal effect inference with deep latent-variable models." Advances in neural information processing systems 30 (2017).
- Hill, Jennifer L. "Bayesian nonparametric modeling for causal inference." Journal of Computational and Graphical Statistics 20.1 (2011): 217-240.


# Development

## Poetry

We use Poetry to manage the project dependencies, they're specified in the [pyproject.toml](pyproject.toml). To install poetry run:

```
    curl -sSL https://install.python-poetry.org | python3 -
```

To install the environment run `poetry install`, this will create a virtualenv that you can use by running either `poetry shell` or `poetry run {command}`. It's also a virtualenv that you can interact with in the normal way too.

More information about poetry can be found [here](https://python-poetry.org/)

## mlflow

We use [mlflow](https://mlflow.org/) for logging metrics and artifacts. By default it will run locally and store results in `./mlruns`.
It is called inside `run_experiment_on_parsed_args` and the target (SQL database etc) can be changed as required.
