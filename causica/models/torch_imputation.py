import numpy as np
import torch
from tqdm import tqdm

from ..models.torch_model import TorchModel
from ..utils.torch_utils import create_dataloader


def impute(model, data, mask, impute_config_dict=None, *, average=True, **vamp_prior_kwargs):
    """
    Fill in unobserved variables using a trained model.

    Processes + minibatches data and passes to impute_processed_batch().

    Data should be provided in unprocessed form, and will be processed before running, and
    will be de-processed before returning (i.e. variables will be in their normal, rather than
    squashed, ranges).

    Args:
        data (numpy array of shape (num_rows, feature_count)): input data in unprocessed form.
        mask (numpy array of shape (num_rows, feature_count)): Corresponding mask, where observed
            values are 1 and unobserved values are 0.
        impute_config_dict (dictionary): Any other parameters needed by a specific concrete class. Of
            the form {arg_name: arg_value}. e.g. {"sample_count": 10}
        average (bool): Whether or not to return the averaged results accross Monte Carlo samples, Defaults to True.
        vamp_prior_kwargs (dict): extra inputs to impute_processed_batch used by specific models, e.g. processed vamp prior data in PVAEBaseModel.

    Returns:
        imputed (numpy array of shape (num_rows, input_dim) or (sample_count, num_rows, input_dim)):
            Input data with missing values filled in, returning averaged or sampled imputations depending
            on whether average=True or average=False.
    """
    if not isinstance(model, TorchModel):
        # This function requires model to implement 'eval' and 'get_device'
        raise NotImplementedError
    model.eval()
    num_rows = data.shape[0]
    assert impute_config_dict is not None
    sample_count = impute_config_dict["sample_count"]
    batch_size = impute_config_dict["batch_size"]

    # Process data.
    processed_data, processed_mask = model.data_processor.process_data_and_masks(data, mask)

    # Create an empty array to store imputed values, with shape (sample_count, batch_size, input_dim)
    # Note that even if using sparse data, we use a dense array here since this array will have all values filled.
    imputed = np.empty((sample_count, *processed_data.shape), dtype=processed_data.dtype)
    with torch.no_grad():
        dataloader = create_dataloader(
            processed_data,
            processed_mask,
            batch_size=batch_size,
            sample_randomly=False,
        )

        for idx, (processed_data_batch, processed_mask_batch) in enumerate(tqdm(dataloader)):
            processed_data_batch = processed_data_batch.to(model.get_device())
            processed_mask_batch = processed_mask_batch.to(model.get_device())
            imputed_batch = model.impute_processed_batch(
                processed_data_batch,
                processed_mask_batch,
                preserve_data=impute_config_dict.get("preserve_data_when_impute", True),
                **impute_config_dict,
                **vamp_prior_kwargs
            )
            idx_start = idx * batch_size
            idx_end = min((idx + 1) * batch_size, num_rows)
            imputed[:, idx_start:idx_end, 0 : model.variables.num_processed_non_aux_cols] = imputed_batch.cpu().numpy()
            imputed[:, idx_start:idx_end, model.variables.num_processed_non_aux_cols :] = (
                processed_data_batch[:, model.variables.num_processed_non_aux_cols :].cpu().numpy()
            )

    # Unprocess data
    unprocessed_imputed_samples = [model.data_processor.revert_data(imputed[i]) for i in range(sample_count)]
    unprocessed_imputed = np.stack(unprocessed_imputed_samples, axis=0)

    # Average non-string samples if necessary
    # For string variables, take the 1st sample as "mean" (as we can't perform mean over string data)
    # TODO #18668: experiment with calculating mean in text embedding space instead
    if average:
        averaged_unprocessed_imputed = np.copy(unprocessed_imputed[0, :, :])
        non_text_idxs = model.variables.non_text_idxs
        averaged_unprocessed_imputed[:, non_text_idxs] = np.mean(unprocessed_imputed[:, :, non_text_idxs], axis=0)
        return averaged_unprocessed_imputed[:, 0 : model.variables.num_unprocessed_non_aux_cols]
    return unprocessed_imputed[:, :, 0 : model.variables.num_unprocessed_non_aux_cols]
