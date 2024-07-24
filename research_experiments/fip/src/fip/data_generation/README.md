# Data Generation

## Structure of the Data
When the data is generated using on the following files, it will create a directory consisting of:
 - `test_x.npy`
 - `train_x.npy`
 - `val_x.npy`
 - `true_graph.npy`
 - `x_cf_i.npy`

 `{split}_x.npy` contains the observaional and noise data concatenated (w.r.t the second dimension) in a numpy array of shape (`num_samples`,`2 * num_nodes`) where `num_samples` represents the size of the dataset, and `num_nodes` the number of 1-dimensional variables. The first `num_nodes` features corresponds to the observational data, while the last `num_nodes` features represent the noise data.

`true_graph.npy` contrains the DAG associated to this dataset. In particular, each row `i` indicates the parents of the `i`-th variable.

`x_cf_i.npy` contains is used to test the counterfactual predictions of models. It contains a numpy array of shape (`num_counterfactual_samples`,`2 * num_nodes + 2`), where the first `num_nodes` features represent the factual data, the next `num_nodes` features represent the counterfactual data, the `2 * num_nodes + 1`-th feature is the index of the intenvention, and the `2 * num_nodes + 2`-th feature is the value of the intervention.
 

## AVICI Data Generation

We support creation of randomly sampled data from [AVICI](https://arxiv.org/abs/2205.12934) with the `avici_data.py` script. Parameters like, type of distributions, total number of seeds and total nodes in the graphs can be manipulated from the script itself.

```
Ex: python -m avici_data --func_type linear --noise_type gaussian --graph_type er --distribution_type in
```

We also provide a bash file  `avici_data.sh` that applies in a grid manner the python file  `avici_data.py`.

```
bash avici_data.sh
```

## CSuite Data Generation

We support creation of continuous [CSuite](https://arxiv.org/abs/2202.02195) datasets with `csuite_data.py` script.

```
Ex: python -m csuite_data --dist_case lingauss
```

We also provide a bash file  `csuite_data.sh` that applies in a grid manner the python file  `csuite_data.py`.

```
bash csuite_data.sh
```

## Causal NF Data Generation

We support creation of [Causal NF](https://arxiv.org/abs/2306.05415) datasets with `normalizing_flow_data.py` script.

```
Ex: python -m normalizing_flow_data --dist_case triangle
```

We also provide a bash file  `normalizing_flow_data.sh` that applies in a grid manner the python file  `normalizing_flow_data.py`.

```
bash normalizing_flow_data.sh
```
