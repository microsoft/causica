# Baseline model for CATE estimation

For a simple baseline for temporal CATE estimation, we provide an adaptation of [DECI](https://arxiv.org/abs/2202.02195), called 
Fold-time DECI (FT-DECI), which supports the temporal data. To use FT-DECI for CATE estimation, one needs to perform the following steps:
- Train FT-DECI with a specific training data in Task 1. The script will save the trained model in a specified folder.
- Load the trained model and conditioning samples, intervention/reference-target pairs, corresponding to the training data,
from the provided json file. Then, compute the CATE estimation for this dataset. 
- Repeat the above for all training datasets in Task 1. Gather all CATE estimations to prepare submission files.

In the following, we give details on each step.
## Train FT-DECI with specific dataset
To train the model, first, clone the Causica repo. Then, place the starting kit at the `<root dir>` of the Causica repo.
It is recommended to use `poetry` to setup the necessary dependencies for the repo. Refer to the repo README.md. 

Copy the task 1 datasets to `<root dir>/data/`. Then, run the following command:
```bash
python -m causica.run_experiment.py <dataset name> -d data/<data folder> --model_type fold_time_deci -dc examples/neurips_competition/starting_kit/task_2/configs/dataset_config_temporal_causal_dataset.json -m examples/neurips_competition/starting_kit/task_2/configs/model_config_fold_time_deci_competition.json -o <output_dir> -dv <gpu or cpu>
```
where `-dv` specifies the device to use ('gpu' or 'cpu'). After training, it will output `model.pt` along with necessary config files to the output directory.

## Estimate CATE for a specific dataset. 
To estimate CATE for a specific dataset, run the following command:
```bash
python -m examples.neurips_competition.starting_kit.task_2.task_2_cate --model_dir <dir saving the model> --model_id <the model id> --data_dir <dir containing task_2 json file> --data_name <json file name> --output_dir <output dir> --device <gpu or cpu>
```
This will output a `cate_estimate_{data_name}.npy` for the specified dataset, with shape `(10)`.
The `model_id` is a string, which can be found in the folder name of the saved model (i.e. model will be saved to `<output_dir>/<time>/models/<model_id>`)

Repeat the above for each dataset in Task 1.

## Preparation for submission
Run the following:
```bash
python -m examples.neurips_competition.starting_kit.task_2.prepare_submission --load_files <path to cate estimate 1> <path to cate estimate 2> ... <path to cate estimate 5> --output_dir <output dir>
```
E.g. 
```bash
python -m examples.neurips_competition.starting_kit.task_2.prepare_submission --load_files output/dataset_1/cate_estimate.npy output/dataset_2/cate_estimate.npy ... output/dataset_5/cate_estimate.npy --output_dir submission/
```
It will output `cate_estimate.npy` with shape `(5,10)`, containing CATE estimations for `5` datasets and `10` query for each dataset.

