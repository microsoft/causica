[![Causica CI Build](https://github.com/microsoft/causica/actions/workflows/ci-build.yml/badge.svg)](https://github.com/microsoft/causica/actions/workflows/ci-build.yml)

# Project Causica

## Overview

Humans make tens of thousands of decisions every day. Project Causica aims to develop machine learning solutions for efficient decision making that demonstrate human expert-level performance across all domains. We develop advanced machine learning solutions in causal discovery, causal inference, and Bayesian experimental design using modern (probabilistic) deep learning methods. Our conceptual framework is to divide decisions into two types: "best next question" and "best next action". 

In daily life, one type of decision we make relates to information gathering for "get to know" decisions; for example, a medical doctor takes a medical test to decide the correct diagnosis for a patient. Humans are very efficient at gathering information and drawing the correct conclusion, while most deep learning methods require significant amounts of training data. Thus, the first part of project Causica focuses on enabling machine learning solutions to gather personalized information, allowing the machine to know the "best next question" and make a final judgment efficiently [1,2,6].
Our technology for "best next question" decisions is driven by state-of-the-art algorithms for Bayesian experimental design and active learning.

The second type of decision made in many domains is intervention, which can be summarised as selecting the "best next action ". For example, a business owner needs to decide which action will lead to greater customer satisfaction; a teacher needs to decide which exercise to give students to help them learn most effectively. Humans are very efficient at making such decisions, primarily due to our causal reasoning abilities. Thus, in project Causica, we develop end-to-end causal inference methods which use existing data to perform  causal discovery and compute causal inference quantities such as (conditional) average treatment effect [8,3]. We not only provide our deep learning-based framework, but we also connect other existing methods such as [DoWhy](https://microsoft.github.io/dowhy/) and [EconML](https://econml.azurewebsites.net/) as alternatives, allowing users to choose the most suitable methods for their applications. 

With these decision-making goals, one can use our codebase in an end-to-end way for decision-making. We also provide the flexibility to use any core functionalities such as missing value prediction, best next question, causal discovery, causal inference, etc, separately depending on the users' needs.     

Our technology has enabled personalized decision-making in real-world systems, combining multiple advanced research methodologies in simple APIs suitable 
for research development in the research community, and commercial use by data scientists and developers. For commercial applications, please reach out to us at  azua-request@microsoft.com  if you are interested in using our technology as a service. 

This project splits causal end to end code from the Azua repo found here [Azua](https://github.com/microsoft/project-azua).

# DECI: End to End Causal Inference

## About

Real-world data-driven decision making requires causal inference to ensure the validity of drawn conclusions. However, it is very uncommon to have a-priori perfect knowledge of the causal relationships underlying relevant variables. DECI allows the end user to perform causal inference without having complete knowledge of the causal graph. This is done by combining the causal discovery and causal inference steps in a single model. DECI takes in observational data and outputs ATE and CATE estimates. 

For more information, please refer to the [paper](https://arxiv.org/abs/2202.02195).


**Model Description**

DECI is a generative model that employs an additive noise structural equation model (ANM-SEM) to capture the functional relationships among variables and exogenous noise, while simultanously learning a variational distribution over causal graphs. Specifically, the relationships among variables are captured with flexible neural networks while the exogenous noise is modelled as either a Gaussian or spline-flow noise model. The SEM is reversible, meaning that we can generate an observation vector from an exogenous noise vector through forward simulation and given a observation vector we can recover a unique corresponding exogenous noise vector. In this sense, the DECI SEM can be seen as a flow from exogenouse noise to observations. We employ a mean-field approximate posterior distribution over graphs, which is learnt together with the functional relationships among variables by optimising an evidence lower bound (ELBO). Additionally, DECI supports learning under partially observed data.

**Simulation-based Causal Inference**

DECI estimates causal quantities (ATE / CATE) by applying the relevant interventions to its learnt causal graph (i.e. mutilating incoming edges to intervened variables) and then sampling from the generative model. This process involves first sampling a vector of exogenous noise from the learnt noise distribution and then forward simulating the SEM until an observation vector is obtained. ATE can be computed via estimating an expectation over the effect variable of interest using MonteCarlo samples of the intervened distribution of observations. DECI can not simulate samples from conditional intervened distributions directly. However, we can learn an estimator of the conditional intervened distrbution from samples of the joint distribution of conditioning variables and effect variables. Specifically, we use radial-basis-function kernel regression to evaluate expectations in the CATE formula.


**Specifying a prior over graphs**

If some a-prior knowledge of the causal graph is available, it can be incorporated as a prior for DECI by using the `informed_deci` model. The a-prior graph should is passed to DECI as a part of the `CausalDataset`. The hyperparameters used to specify strenght of beleif in the prior graph are discussed below. 


**Specifying a noise model**

If the exogenous noise is known to be Gaussian, we reccomend employing the `deci_gaussian` model. This model has its Gaussian exogenous noise distribution mean set to 0 while its variance is learnt.

In the more general setting, we reccomend using `deci_spline`. Here the exogenous noise distribution is a flexible spline flow that is learnt from the data. This model provides most gains in heavy-tailed noise settings, where the Gaussian model is at risk of overfitting to outliers. 

## How to run

The command to train this model with the csuite_symprod_simpson dataset and evaluate causal discovery and (C)ATE estimation performance is:

```
python run_experiment.py csuite_symprod_simpson --model_type deci_spline --model_config configs/deci/true_graph_deci_spline.json -dc configs/dataset_config_causal_dataset.json -c -te
```

## How to generate data for experiments

We use different data for each of our 3 experiments: large scale evluation data, csuite synthetic data and Twins/IHDP

**Large scale evaluation**

There's three different types of data we use:
1. Synthetic. Samples a random graph with N nodes and E edges and generates the data using gaussian processes for the functions in the SEM. Go to `causica/data_generation/large_synthetic` and run `python data_generation.py`. This will store the datasets generated in the `causica/data_generation/large_synthetic/data/` directory.
2. Pseudo-real. Syntren, a simulator that creates synthetic transcriptional regulatory networks and produces simulated gene expression data that approximates experimental data. To generate the data open the file `causica/data_generation/syntren/generate.py` and follow the instructions within.
3. Real. Protein cells datasets, from Sachs et. al. 2005. The data measures the expression level of different proteins and phospholipids in human cells. To generate the data open the file `causica/data_generation/protein_cells/generate.py` and follow the instructions within.

**CSuite**

The CSuite datasets can generated by running `causica/data_generation/csuite/simulate.py` from the top-level directory of the package.
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
python run_experiment.py ihdp_norm --model_type deci_gaussian --model_config configs/deci/deci_gaussian.json -te -dc configs/dataset_config_causal_dataset.json 
```

* DECI Spline on IHDP for ATE estimation:

```
python run_experiment.py ihdp_norm --model_type deci_spline --model_config configs/deci/deci_spline.json -te -dc configs/dataset_config_causal_dataset.json 
```

* True graph DECI Gassian on IHDP for ATE estimation:

```
python run_experiment.py ihdp_norm --model_type deci_gaussian --model_config configs/deci/true_graph_deci_gaussian.json -te -dc configs/dataset_config_causal_dataset.json 
```

* True graph DECI Spline on IHDP for ATE estimation:

```
python run_experiment.py ihdp_norm --model_type deci_spline --model_config configs/deci/true_graph_deci_spline.json -te -dc configs/dataset_config_causal_dataset.json 
```

* DECI Gaussian DoWhy linear on IHDP for ATE estimation:

```
python run_experiment.py ihdp_norm --model_type deci_dowhy --model_config configs/deci/deci_gaussian_dowhy_linear.json -te -dc configs/dataset_config_causal_dataset.json 
```

* DECI Spline DoWhy linear on IHDP for ATE estimation:

```
python run_experiment.py ihdp_norm --model_type deci_dowhy --model_config configs/deci/deci_spline_dowhy_linear.json -te -dc configs/dataset_config_causal_dataset.json
```

* True graph DoWhy linear on IHDP for ATE estimation:

```
python run_experiment.py ihdp_norm --model_type true_graph_dowhy --model_config configs/deci/dowhy_linear.json -te -dc configs/dataset_config_causal_dataset.json
```


* DECI Gaussian DoWhy nonlinear on IHDP for ATE estimation:

```
python run_experiment.py ihdp_norm --model_type deci_dowhy --model_config configs/deci/deci_gaussian_dowhy_nonlinear.json -te -dc configs/dataset_config_causal_dataset.json
```

* DECI Spline DoWhy nonlinear on IHDP for ATE estimation:

```
python run_experiment.py ihdp_norm --model_type deci_dowhy --model_config configs/deci/deci_spline_dowhy_nonlinear.json -te -dc configs/dataset_config_causal_dataset.json
```

* True graph DoWhy nonlinear on IHDP for ATE estimation:

```
python run_experiment.py ihdp_norm --model_type true_graph_dowhy --model_config configs/deci/dowhy_nonlinear.json -te -dc configs/dataset_config_causal_dataset.json
```

* PC DoWhy linear on IHDP for ATE estimation:

```
python run_experiment.py ihdp_norm --model_type pc_dowhy --model_config configs/deci/pc_dowhy_linear.json -te -dc configs/dataset_config_causal_dataset.json
```


* PC DoWhy nonlinear on IHDP for ATE estimation:

```
python run_experiment.py ihdp_norm --model_type pc_dowhy --model_config configs/deci/pc_dowhy_nonlinear.json -te -dc configs/dataset_config_causal_dataset.json
```

### CSuite

To launch continuous CSuite experiments, use a command of the form

```
python run_experiment.py <dataset> --model_type <model_type> --model_config <model_config> -c -te -dc configs/dataset_config_causal_dataset.json 
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
python run_experiment.py <dataset> --model_type <model_type> --model_config <model_config> -c <-te> -dc configs/dataset_config_causal_dataset.json 
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
python run_experiment.py SF_64_64_spline_sem_mlp_noise_4_seed --model_type deci_dowhy --model_config configs/deci/deci_spline_dowhy_nonlinear.json -c -te -dc configs/dataset_config_causal_dataset.json 
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
- `cate_rff_n_features`: number of random features to use in functiona approximation when estimating CATE. 
- `cate_rff_lengthscale`: lengthscale of RBF kernel used when estimating CATE.
- `prior_A`: prior adjacency matrix. Does not need to be a DAG.
- `prior_A_confidence`: degree of confidence in prior adjacency matrix enabled edges between 0 and 1. e.g. a values 0.5 enforces a beleif that the entries in prior_A are correct 50% of the time.

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

## References

If you have used the models in our code base, please consider to cite the corresponding papers:

[1], **(VISL)** Pablo Morales-Alvarez, Wenbo Gong, Angus Lamb, Simon Woodhead, Simon Peyton Jones, Nick Pawlowski, Miltiadis Allamanis, Cheng Zhang, "Simultaneous Missing Value Imputation and Structure Learning with Groups", [ArXiv preprint](https://arxiv.org/abs/2110.08223)

[2], **(DECI)** Tomas Geffner, Javier Antoran, Adam Foster, Wenbo Gong, Chao Ma, Emre Kiciman, Amit Sharma, Angus Lamb, Martin Kukla, Nick Pawlowski, Miltiadis Allamanis, Cheng Zhang. Deep End-to-end Causal Inference. [Arxiv preprint](https://arxiv.org/abs/2202.02195) (2022)


Other references:
- Louizos, Christos, et al. "Causal effect inference with deep latent-variable models." Advances in neural information processing systems 30 (2017).
- Hill, Jennifer L. "Bayesian nonparametric modeling for causal inference." Journal of Computational and Graphical Statistics 20.1 (2011): 217-240.