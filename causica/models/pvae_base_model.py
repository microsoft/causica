# This is required in python 3 to allow return types of the same class.
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple, Union, cast

import numpy as np
import torch
from scipy.sparse import csr_matrix, issparse
from tqdm import tqdm

from ..datasets.dataset import Dataset, SparseDataset
from ..datasets.variables import Variables
from ..objectives.eddi import EDDIObjective
from ..utils.data_mask_utils import restore_preserved_values, sample_inducing_points
from ..utils.helper_functions import to_tensors
from ..utils.torch_utils import create_dataloader
from ..utils.training_objectives import gaussian_negative_log_likelihood, negative_log_likelihood
from .imodel import IModelForObjective
from .torch_imputation import impute
from .torch_model import TorchModel
from .torch_training_types import LossConfig

EPSILON = 1e-5


class PVAEBaseModel(TorchModel, IModelForObjective):
    """
    Abstract model class.

    To instantiate this class, these functions need to be implemented:
        _train: Run the training loop for the model.
        _impute: Fill in any missing values for test data.
        _reconstruct: Reconstruct data by passing them through the VAE
        name: Name of model implementation.
    """

    def __init__(self, model_id: str, variables: Variables, save_dir: str, device: torch.device) -> None:
        """
        Args:
            model_id: Unique model ID for referencing this model instance.
            variables: Information about variables/features used by this model.
            save_dir: Location to save any information about this model, including training data.
                It will be created if it doesn't exist.
            device: Name of Torch device to create the model on. Valid options are 'cpu', 'gpu', or a device ID
                (e.g. 0 or 1 on a two-GPU machine).
        """
        super().__init__(model_id, variables, save_dir, device)
        self._alpha = 1.0  # The default value for the categorical likelihood coefficient.

    @staticmethod
    def _split_vamp_prior_config(training_config: Dict[str, Any]) -> Tuple[dict, dict]:
        # Split training config into (training_config, vamp_prior_config)
        training_config = training_config.copy()
        vamp_prior_config = {"save_vamp_prior": training_config.pop("save_vamp_prior")}
        for k in ["vamp_prior_reward_samples", "vamp_prior_inducing_points"]:
            vamp_prior_config.update({k: training_config.pop(k, None)})
        return training_config, vamp_prior_config

    def _save_vamp_prior(
        self,
        processed_dataset: Union[Dataset, SparseDataset],
        save_vamp_prior: bool,
        vamp_prior_inducing_points: Optional[int] = None,
        vamp_prior_reward_samples: Optional[int] = None,
    ) -> None:
        if not save_vamp_prior:
            return
        assert vamp_prior_inducing_points is not None
        assert vamp_prior_reward_samples is not None
        train_data, train_mask = processed_dataset.train_data_and_mask
        vamp_prior_data = sample_inducing_points(train_data, train_mask, row_count=vamp_prior_inducing_points)
        vamp_prior_data = cast(Union[Tuple[np.ndarray, np.ndarray], Tuple[csr_matrix, csr_matrix]], vamp_prior_data)
        EDDIObjective.calc_and_save_vamp_prior_info_gain(self, vamp_prior_data, sample_count=vamp_prior_reward_samples)

    def run_train(
        self,
        dataset: Union[Dataset, SparseDataset],
        train_config_dict: Optional[Dict[str, Any]] = None,
        report_progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:

        """
        Train the model.
        Training results will be saved.

        Args:
            dataset: Dataset object with data and masks in unprocessed form.
            train_config_dict (dictionary): Any other parameters needed by a specific concrete class. Of
                the form {arg_name: arg_value}. e.g. {"learning_rate": 1e-3, "epochs": 100}
            report_progress_callback: Function to report model progress for API.
        """
        if train_config_dict is None:
            train_config_dict = {}
        train_config_dict, vamp_prior_config = self._split_vamp_prior_config(train_config_dict)
        processed_dataset = self.data_processor.process_dataset(dataset)
        self._train(
            dataset=processed_dataset,
            report_progress_callback=report_progress_callback,
            **train_config_dict,
        )
        self._save_vamp_prior(processed_dataset, **vamp_prior_config)

    @abstractmethod
    def _train(self, *args, **kwargs):
        pass

    def impute(self, data, mask, impute_config_dict=None, *, vamp_prior_data=None, average=True):
        if vamp_prior_data is None:
            return impute(self, data, mask, impute_config_dict=impute_config_dict, average=average)
        else:
            processed_vamp_data_array = self.data_processor.process_data_and_masks(*vamp_prior_data)
            # Keep processed VampPrior data on CPU until we sample inducing points, as this data can be large and is
            # not required for any CUDA computations.
            return impute(
                self,
                data,
                mask,
                impute_config_dict=impute_config_dict,
                average=average,
                vamp_prior_data=to_tensors(*processed_vamp_data_array, device=torch.device("cpu")),
            )

    def impute_processed_batch(
        self: PVAEBaseModel,
        data: torch.Tensor,
        mask: torch.Tensor,
        *,
        sample_count: int,
        preserve_data: bool = True,
        vamp_prior_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
         Fill in unobserved variables in a minibatch of data using a trained model. Optionally, use a vamp prior to
         impute empty rows, and optionally replace imputed values with input values for observed features.

         Assumes data is a torch.Tensor and in processed form (i.e. variables will be in their squashed ranges,
         and categorical variables will be in one-hot form).

         Args:
             data (shape (batch_size, input_dim)): Data to be used to train the model, in processed form.
             mask (shape (batch_size, input_dim)): Data observation mask, where observed values are 1 and unobserved
                 values are 0.
             sample_count: Number of imputation samples to generate.
             vamp_prior_data (Tuple of (torch tensor, torch tensor)): Data to be used to fill variables if using the VAMP
                 prior method. Format: (data, mask). This defaults to None, in which case the VAMP Prior method will not
                 be used.
             preserve_data (bool): Whether or not to impute data already present. Defaults to True, which keeps data
                 present in input.

        Returns:
             imputations (torch.Tensor of shape (sample_count, batch_size, output_dim)): Input data with missing values
                 filled in.
        """
        if not isinstance(data, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise ValueError("data and mask should be tensors. To work on ndarrays, use impute")
        assert data.shape == mask.shape
        assert data.shape[1] == self.input_dim
        batch_size, num_features = data.shape
        if self.variables.has_auxiliary:
            num_features = self.variables.num_processed_non_aux_cols

        imputations = torch.full((sample_count, batch_size, num_features), np.nan, device=self.device)

        # vamp_rows are rows where input is completely unobserved
        vamp_rows = torch.where(mask.sum(dim=1) == 0)[0]
        if vamp_prior_data is not None and vamp_rows.numel() > 0:
            imputed_from_vamp = self._impute_from_vamp_prior(sample_count * vamp_rows.numel(), vamp_prior_data)
            imputed_from_vamp = imputed_from_vamp.reshape(sample_count, vamp_rows.numel(), -1)
            imputations[:, vamp_rows, :] = imputed_from_vamp

            not_vamp_rows = torch.where(mask.sum(dim=1) != 0)[0]

        else:
            not_vamp_rows = torch.arange(batch_size)

        if len(not_vamp_rows) > 0:
            not_vamp_data = data[not_vamp_rows]
            not_vamp_mask = mask[not_vamp_rows]
            imputed_not_vamp_data = self._reconstruct_and_reshape(
                not_vamp_data, not_vamp_mask, sample_count=sample_count, **kwargs
            )
            imputations[:, not_vamp_rows, :] = imputed_not_vamp_data

        if preserve_data:
            imputations = restore_preserved_values(self.variables, data, imputations, mask)
        return imputations

    def get_model_pll(
        self: PVAEBaseModel,
        data: np.ndarray,
        feature_mask: np.ndarray,
        target_idx,
        sample_count: int = 50,
    ):
        """
        Computes the predictive log-likelihood of the target-data given the feature_mask-masked data as input.

        Args:
            data (Numpy array of shape (batch_size, feature_count)): Data in unprocessed form to be used to
                compute the pll.
            feature_mask (Numpy array of shape (batch_size, feature_count)): Mask indicating conditioning
                variables for computing the predictive log-likelihood.
            target_idx (int): Column index of target variable for compute the likelihood of.
            sample_count (int): Number of Monte Carlo samples to use from the latent space. Defaults to 50.

        Returns:
            predictive_ll (float): Mean predictive log-likelihood (mean taken over batch dim in data).

        """
        # Process input data
        (
            proc_feature_data_array,
            proc_feature_mask_array,
        ) = self.data_processor.process_data_and_masks(data, feature_mask)
        proc_feature_data, proc_feature_mask = to_tensors(
            proc_feature_data_array, proc_feature_mask_array, device=self.device
        )

        # Create target_mask from target_index
        target_mask = np.zeros_like(data, dtype=bool)
        target_mask[:, target_idx] = 1

        # Process target data
        (
            proc_target_data_array,
            proc_target_mask_array,
        ) = self.data_processor.process_data_and_masks(data, target_mask)
        proc_target_data, proc_target_mask = to_tensors(
            proc_target_data_array, proc_target_mask_array, device=self.device
        )

        # Expand target data and mask to be shape (sample_count, batch_size, feature_count)
        proc_target_data = proc_target_data.expand(sample_count, *proc_target_data.shape)
        proc_target_mask = proc_target_mask.expand(sample_count, *proc_target_mask.shape)

        # Compute PVAE outputs given input features (parameters of the Gaussian mixture)
        (dec_mean, dec_logvar), _, _ = self.reconstruct(proc_feature_data, proc_feature_mask, count=sample_count)

        # Compute Gaussian negative log-likelihood per sample in sample_count
        gnll = gaussian_negative_log_likelihood(
            proc_target_data, dec_mean, dec_logvar, mask=proc_target_mask, sum_type=None
        )
        gnll = gnll[:, :, target_idx]
        predictive_ll = -gnll
        predictive_ll = torch.logsumexp(predictive_ll, dim=0) - np.log(sample_count)
        predictive_ll = predictive_ll.mean()

        return predictive_ll

    def get_marginal_log_likelihood(
        self,
        impute_config: Dict[str, int],
        data: Union[np.ndarray, csr_matrix],
        observed_mask: Optional[Union[np.ndarray, csr_matrix]] = None,
        target_mask: Optional[Union[np.ndarray, csr_matrix]] = None,
        evaluate_imputation: Optional[bool] = False,
        num_importance_samples: int = 5000,
        **kwargs,
    ) -> float:
        """
        Estimate marginal log-likelihood of the data using importance sampling:
        - Imputation MLL -> imputed data given the observed data log p(x_u|x_o) if evaluate_imputation is True
        - Reconstruction MLL -> all data log p(x) otherwise

        Args:
            impute_config: Dictionary containing options for inference.
            data: Data in unprocessed form to be used with shape (num_rows, input_dim).
            mask: If not None, mask indicating observed variables with shape (num_rows, input_dim). 1 is observed,
                  0 is un-observed. If None everything is marked as observed.
            target_mask: Values masked during imputation to use as prediction targets, where 1 is a target, 0 is not.
                If None, nothing is marked as an imputation target.
            evaluate_imputation: Whether to estimate Imputation MLL log p(x_u|x_o) or Reconstruction MLL log p(x).
            num_importance_samples: The number of importance samples to be taken.
            **kwargs: Extra keyword arguments required by reconstruct.
        Returns:
            marginal_log_likelihood: The estimated marginal log likelihood averaged across data points.
        """
        # TODO(17895): Add Generation MLL option to the marginal log-likelihood metric.

        batch_size = impute_config["batch_size"]

        # Assumed to only work on dense arrays for now
        if issparse(data):
            data = cast(csr_matrix, data)
            data = data.toarray()
        if issparse(observed_mask):
            observed_mask = cast(csr_matrix, observed_mask)
            observed_mask = observed_mask.toarray()
        if issparse(target_mask):
            target_mask = cast(csr_matrix, target_mask)
            target_mask = target_mask.toarray()
        if observed_mask is None:
            observed_mask = np.ones_like(data, dtype=bool)
        if target_mask is None:
            assert not evaluate_imputation
            target_mask = np.zeros_like(data, dtype=bool)
        assert data.shape == observed_mask.shape
        assert data.shape == target_mask.shape

        num_rows, _ = data.shape

        # TODO(17896): Add processing and batching of extra data objects
        processed_data, processed_obs_mask, processed_target_mask = self.data_processor.process_data_and_masks(
            data, observed_mask, target_mask
        )
        marginal_log_likelihood = np.empty((num_rows,), dtype=processed_data.dtype)

        with torch.no_grad():
            dataloader = create_dataloader(
                processed_data,
                processed_obs_mask,
                processed_target_mask,
                batch_size=batch_size,
                sample_randomly=False,
            )

            for idx, (processed_data_batch, processed_obs_mask_batch, processed_target_mask_batch) in enumerate(
                tqdm(dataloader)
            ):
                processed_data_batch = processed_data_batch.to(self.device)
                processed_obs_mask_batch = processed_obs_mask_batch.to(self.device)
                processed_target_mask_batch = processed_target_mask_batch.to(self.device)

                log_importance_weights = self._get_log_importance_weights(
                    processed_data_batch,
                    processed_obs_mask_batch,
                    processed_target_mask_batch,
                    evaluate_imputation=cast(bool, evaluate_imputation),
                    num_importance_samples=num_importance_samples,
                    **kwargs,
                )  # Shape (num_importance_samples, batch_size)
                average_factor = torch.log(torch.tensor(num_importance_samples, dtype=torch.float))
                marginal_log_likelihood_batch = (
                    torch.logsumexp(log_importance_weights, dim=0) - average_factor
                )  # Shape (batch_size,)

                idx_start = idx * batch_size
                idx_end = min((idx + 1) * batch_size, num_rows)
                marginal_log_likelihood[idx_start:idx_end] = marginal_log_likelihood_batch.cpu().numpy()

        return marginal_log_likelihood.sum().item() / num_rows

    @abstractmethod
    def reconstruct(
        self, data: torch.Tensor, mask: Optional[torch.Tensor], sample: bool = True, count: int = 1, **kwargs: Any
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, Tuple[torch.Tensor, torch.Tensor],]:
        """
        Reconstruct data by passing them through the VAE.
        Note that mask=None should always be used in subclasses that don's support missing values.

        Args:
            data: Input data with shape (batch_size, input_dim).
            mask: If not None, mask indicating observed variables with shape (batch_size, input_dim). 1 is observed,
                  0 is un-observed.
            sample: If True, samples the latent variables, otherwise uses the mean.
            count: Number of samples to reconstruct.
            **kwargs: Extra keyword arguments required.

        Returns:
            (decoder_mean, decoder_logvar): Reconstucted variables, output from the decoder. Both are shape (count, batch_size, output_dim). Count dim is removed if 1.
            samples: Latent variable used to create reconstruction (input to the decoder). Shape (count, batch_size, latent_dim). Count dim is removed if 1.
            (encoder_mean, encoder_logvar): Output of the encoder. Both are shape (batch_size, latent_dim)
        """
        raise NotImplementedError()

    def validate_loss_config(self, loss_config: LossConfig) -> None:
        assert loss_config.score_imputation is not None and loss_config.score_reconstruction is not None
        assert loss_config.score_reconstruction or loss_config.score_imputation
        assert loss_config.max_p_train_dropout is not None

    def _impute_from_vamp_prior(
        self, num_samples: int, vamp_prior_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        vp_data, vp_mask = vamp_prior_data
        assert vp_data.shape == vp_mask.shape
        assert vp_data.shape[1] == self.variables.num_processed_cols
        # Sample inducing points for all rows, shape (sample_count * num_vamp_rows, input_dim)
        inducing_data, inducing_mask = sample_inducing_points(vp_data, vp_mask, num_samples)
        # Only move to GPU once we have sampled the inducing points as these tensors are much smaller.
        inducing_data, inducing_mask = (
            inducing_data.to(self.device),
            inducing_mask.to(self.device),
        )
        # Shape (1, num_samples, output_dim)
        return self._reconstruct_and_reshape(inducing_data, inducing_mask, sample_count=1)

    def _reconstruct_and_reshape(
        self, data: torch.Tensor, mask: Optional[torch.Tensor], sample_count: int, **kwargs
    ) -> torch.Tensor:
        """
        Make sample_count imputations of missing data for given data and mask.

         Args:
            data: partially observed data with shape (batch_size, input_dim).
            mask: mask indicating observed variables with shape (batch_size, input_dim). 1 is observed, 0 is
                un-observed.
            If None, will be set to all 1's.
            sample_count: Number of samples to take.

        Returns:
            imputations: PyTorch Tensor with shape: (sample_count, batch_size, input_dim)
        """
        if mask is None:
            mask = torch.ones_like(data)
        assert data.dim() == 2
        assert mask.shape == data.shape
        assert data.shape[1] == self.variables.num_processed_cols
        (imputations, _), _, _ = self.reconstruct(data=data, mask=mask, sample=True, count=sample_count, **kwargs)
        if self.variables.has_auxiliary:
            data = data[:, 0 : self.variables.num_processed_non_aux_cols]
        return imputations.reshape(sample_count, *data.shape)

    def _get_log_importance_weights(
        self,
        data: torch.Tensor,
        observed_mask: torch.Tensor,
        target_mask: torch.Tensor,
        evaluate_imputation: bool,
        num_importance_samples: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate a set of log importance weights/samples to estimate marginal log-likelihood.
        Collect samples from z ~ q(z|x) to estimate:
        - Imputation MLL -> return log [p(x_u|z) q(z|x_o) / q(z|x)] if evaluate_imputation is True
        - Reconstruction MLL -> return log [p(x|z) p(z) / q(z|x)] otherwise

        This function assumes that latent prior distribution is standard Normal p(z) ~ N(0, 1).

        Args:
            data: Data to be used with shape (batch_size, input_dim).
            mask: Mask indicating observed values in data with shape (batch_size, input_dim). 1 is observed,
                  0 is un-observed.
            target_mask: Values marked as prediction targets during imputation, where 1 is a target and 0 is not.
            evaluate_imputation: Whether to collect samples for Imputation MLL log p(x_u|x_o) or Reconstruction MLL log p(x).
            num_importance_samples: The number of importance samples to be taken.
            **kwargs: Extra keyword arguments required by reconstruct.

        Returns:
            log_importance_weights: Log of importance weights with shape (num_importance_samples, batch_size).
        """
        assert observed_mask is not None
        assert target_mask is not None
        assert data.shape == observed_mask.shape
        assert data.shape == target_mask.shape

        data_non_aux = data[:, 0 : self.variables.num_processed_non_aux_cols]
        num_non_aux_vars = self.variables.num_unprocessed_non_aux_cols
        batch_size, _ = data.shape

        # Collect samples
        (dec_mean, dec_logvar), latent_samples, (enc_mean, enc_logvar) = self.reconstruct(
            data=data, mask=observed_mask, sample=True, count=num_importance_samples, **kwargs
        )
        latent_samples = latent_samples.reshape(num_importance_samples, batch_size, -1)

        # Calculate nll i.e. -log[p(x_u|z)] or -log[p(x|z)]
        if evaluate_imputation:
            mask_nll = target_mask
        else:
            mask_nll = observed_mask

        nll = negative_log_likelihood(
            data=data_non_aux.repeat(num_importance_samples, 1),
            decoder_mean=dec_mean.reshape(
                num_importance_samples * batch_size, self.variables.num_processed_non_aux_cols
            ),
            decoder_logvar=dec_logvar.reshape(
                num_importance_samples * batch_size, self.variables.num_processed_non_aux_cols
            ),
            variables=self.variables,
            alpha=self._alpha,
            mask=mask_nll.repeat(num_importance_samples, 1),
            sum_type=None,
        )  # Shape (num_importance_samples * batch_size, num_non_aux_vars)
        nll = nll.reshape(
            num_importance_samples, batch_size, num_non_aux_vars
        )  # Shape (num_importance_samples, batch_size, num_non_aux_vars
        nll = nll.sum(dim=2)  # Shape (num_importance_samples, batch_size)

        # Calculate log latent variational log[q(z|x)]
        log_latent_variational = (-1) * gaussian_negative_log_likelihood(
            targets=latent_samples, mean=enc_mean, logvar=enc_logvar, mask=None, sum_type=None
        )  # Shape (num_importance_samples, batch_size, latent_dim)
        log_latent_variational = log_latent_variational.sum(axis=2)  # Shape (num_importance_samples, batch_size)

        # Calculate log latent prior log[q(z|x_o)] or log[p(z)]
        if evaluate_imputation:
            (_, _), _, (latent_prior_mean, latent_prior_logvar) = self.reconstruct(
                data=data, mask=observed_mask, sample=False, count=1, **kwargs
            )
        else:
            latent_prior_mean = torch.tensor(0.0)
            latent_prior_logvar = torch.log(torch.tensor(1.0))

        log_latent_prior = (-1) * gaussian_negative_log_likelihood(
            targets=latent_samples,
            mean=latent_prior_mean,
            logvar=latent_prior_logvar,
            mask=None,
            sum_type=None,
        )  # Shape (num_importance_samples, batch_size, latent_dim)
        log_latent_prior = log_latent_prior.sum(axis=2)  # Shape (num_importance_samples, batch_size)

        # Calculate log importance weights
        log_importance_weights = (
            (-1) * nll + log_latent_prior - log_latent_variational
        )  # Shape (num_importance_samples, batch_size)
        return log_importance_weights
