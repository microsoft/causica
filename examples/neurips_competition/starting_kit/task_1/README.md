# Starting Kit for Causal Discovery

As a simple baseline for causal discovery, we provide an interface for using [VARLiNGaM](https://lingam.readthedocs.io/en/latest/tutorial/var.html) on the observational data.

To run the model, copy the datasets into the `<root dir>/data/<dataset_name>` directory and then call:

```bash
python -m causica.run_experiment.py <dataset_name> --model_type varlingam -dc examples/neurips_competition/starting_kit/task_1/dataset_config.json
```

This loads the specified dataset `<dataset_name>` runs VARLiNGaM and saves the model. After training multiple of these models and having them saved to `<save_dir_n>` you can use the example submission preparation script `prepare_submission.py`:
```
python -m examples.neurips_competition.starting_kit.task_1.prepare_submission -l <save_dir_1> ... <save_dir_n> -o <output_file>
```
which loads the model for every dataset and stacks the adjacency matrices. This is saved to `<output_file>` (which defaults to `adj_matrices.npy`) and is ready for submission. Specifically, the evaluation script requires the submission file to be called `adj_matrices.npy`.