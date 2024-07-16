# Minimal Lightning Example

This is an example of how to implement a custom project using the causica package and its PyTorch Lightning interface.
This example is based upon the [Sales Notebook](../multi_investment_sales_attribution.ipynb).

You can run the example code using the following:

```bash
PYTHONPATH="." python -m causica.lightning.main \
--config config/training.yaml \
--data config/data.yaml \
--model config/model.yaml \
--trainer.accelerator gpu
```

## Code:
The code is structured as follows:

- config: Lightning related configs
    - data.yaml: Config related to the dataloader
    - model.yaml: Config related to the model parameters
    - training.yaml: Config related to the training parameters
- data_module.py: The data module for the Lightning model. This loads the data from storage, splits it into train, validation, and test set -- and lastly, creates the dataloaders.
- lightning_module.py: The Lightning module that defines the model, loss, and optimizer used for training.
- model_analysis.ipynb: Sample notebook to analyze the trained model.
