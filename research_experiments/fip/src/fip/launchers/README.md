# How to Use Launcher Files

The launcher files allows to run command of the form:
```console
python -m fip.entrypoint 
--config <path_the_model_config>
--data <path_to_the_data_config>
```
specific for each type of models. There are three launcher files: (i) a file to launch the training of the TO inference method, (ii) a file to launch the training of FiP when (partial) causal knowledge is known, and (iii) another file to launch the training of FiP end-to-end. The latter assume that a TO inference method has been trained beforehand.


### Zero-Shot Inference of TOs
To train the TO inference method, run the following command:
```console
    python -m fip.launchers.amortization
```

### Learn FiP with (Partial) Causal Knowledge
To train FiP when the DAG is known, run the following command:
```console
    python -m fip.launchers.scm_learning_with_ground_truth 
    --ground_truth_case graph 
    --standardize 
```
If you want to train FiP, when only the TO is known, replace  `--ground_truth_case graph` with `--ground_truth_case perm`.


### Learn FiP End-to-End
To train FiP end-to-end, run the following command:
```console
    python -m fip.launchers.scm_learning_with_predicted_truth
    --run_id <name_of_the_directory_containing_the_amortized_model>
    --standardize 
```
This command assumes that a TO inference model has been trained and saved in a directory located at `src/fip/outputs/<name_of_the_directory_containing_the_amortized_model>`.


### Specific Arguments for FiP
To train FiP, either with (partial) causal knowledge, or end-to-end, you can precise some dataset arguments such as:
 - the type dataset on which FiP will be trained by adding: `--dist_case <name_of_dataset>` where `<name_of_dataset>` is the name of the directory containing the data and which has the following location `src/fip/data/<name_of_dataset>`
 - the dimension of the problem by adding: `--total_nodes <dimension>` where `<dimension>` is the number of 1 dimensional nodes to consider


## Resume a Job Locally
If you want to continue the training of a model add the argument: `--local_resume`.
Also, you need to add another argument to provide the name of the directory containing the current model: `--run_name_resume <name_of_the_directory_containing_the_model>`. This command assumes that a model has been saved in a directory located at `src/fip/outputs/<name_of_the_directory_containing_the_model>`.

## Distributed Argument
If you want to parallelize the training you need to add the following arguments: --instance_count (>= 1), --gpus_per_node (>= 1)
.  

## Alternative Way to Launch Jobs
Otherwise, you can build your own job by running:
```console
python -m fip.entrypoint 
--config <path_the_model_config>
--data <path_to_the_data_config> 
```
where:
- `<path_the_model_config>` is the config path of the model that needs to be trained.
- `<path_the_data_config>` is the config path of the data module on which the model will be trained.

Note that to train a FiP model, one needs to use `src/fip/config/numpy_tensor_data_module.yaml` as the config file for the data module. 
While to train a TO inference method,  one needs to use `src/fip/config/synthetic_data_module.yaml` as the config file for the data module. 


