# Baseline model for task 4 CATE estimation 
We utilize the same baseline model for this task as in task_2. The main issue is that the raw training data
cannot be directly used. Thus, one need to perform the following steps:
- Process the raw training data into the same format as the synthetic data in task 2.
- Train the model on the processed data.
- Estimate the corresponding cate according to the provided questionnaire.

All the following steps require setting up the necessary dependencies using poetry.  
## Process the raw training data
To process the training data, one can run
```bash
python -m examples.neurips_competition.starting_kit.task_4.task_4_process_data --data_path <path to the training csv file> --save_dir <dir of the processed training data>
```
This will output 2 files: (1) `<save_dir>/train.csv`, which is the processed training data; 
(2) `<save_dir>/const_map.npy`, which contains the mapping from the raw construct id and the new id used in the processed data.

We made several assumptions regarding the data processing, which may not be optimal. The candidates should NOT rely on this script for data processing.
This is only used as a starting kit. The assumptions:
- The processed data should have the same data format as the synthetic data. 
- For missing construct knowledge, we use zero-imputing by filling a constant value 0.25.
- We assume everytime a checkin question is answered incorrectly, a learning session happens (i.e. a lesson or a hint). 
Due to the fixed procedure, other type (i.e. CheckinRetry, Checkout, CheckoutRetry) should teach the same construct as the Checkin. 
Therefore, we aggregated them into one learning session. 
- We assume we a question is answered correctly, no learning happens. It only reveals the construct knowledge at the current time step. 


## Train the model
This is similar to task 2, one can train the model by running:
```bash
python -m causica.run_experiment <dataset_name> -d <dir that contains <dataset_name>> --model_type fold_time_deci 
-dc examples/neurips_competition/starting_kit/task_4/configs/dataset_config_temporal_causal_dataset.json -m examples/neurips_competition/starting_kit/task_4/configs/model_config_fold_time_deci_competition.json -o <output_dir> -dv <gpu or cpu>
```
The <dataset_name> is the folder that contains the processed training data, and <dir that contains <dataset_name>> is the parent folder that contains <dataset_name>.
For example, if the processed data is stored at `data/A/processed_dataset`, then <dataset_name> is `processed_dataset` and <dir that contains <dataset_name>> is `data/A`.
<output_dir> is the folder that contains the saved training model. The saved model file is at `<output_dir>/<timestamp>/models/<model_id>/`

## Prepare adjancency matrix for submission

After training the model you can use the following script to sample multiple adjancency matrices from FT-DECI, convert them to an aggregated temporal adjacency matrix and stack them to confirm to the submission format
```
python -m examples.neurips_competition.starting_kit.task_3_and_4.task_3_prepare_submission -mt ft_deci -l <save_dir> -o <output_file> -m <path_to_const_map>/const_map.npy -r <path_to_request_file>/constructs_input_test.csv -n <number_of_posterior_samples>
```
which loads the model for every dataset and stacks the adjacency matrices. This is saved to `<output_file>` (which defaults to `adj_matrices.npy`) and is ready for submission. Specifically, the evaluation script requires the submission file to be called `adj_matrices.npy`.

## Estimate CATE for submission
After training the model, one can estimate the CATE by running:
```bash
python -m examples.neurips_competition.starting_kit.task_4.task_4_cate --model_dir <dir contains the saved model> --model_id <model_id> --question_path <the file path to the provided questionnaire>
--contruct_map <file path to the construct map from the processing step> --train_data <file path to the processed train.csv>
--output_dir <dir to save the cate estimate npy file>
--device <gpu or cpu>
```
The `--model_dir` is `<output_dir>/<timestamp>/models/` from the model training step, `--model_id` can be found in that folder, `--question_path` is the file path to the provided 
questionnaire for public leaderboard, `--construct_map` is the file path to the construct map from the processing step, `--train_data` is the file path to the processed training data, 
`--output_dir` is the dir to save the cate estimate npy file, and `--device` is the device to use.

The output file is `cate_estimate.npy` with shape `(num_queries in questionnaire)`.
