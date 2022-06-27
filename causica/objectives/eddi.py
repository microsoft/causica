import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse

from ..models.imodel import IModelForObjective
from ..utils.data_mask_utils import add_to_data, add_to_mask
from ..utils.helper_functions import to_tensors
from ..utils.io_utils import read_json_as, save_json
from ..utils.torch_utils import create_dataloader
from ..utils.training_objectives import kl_divergence


class EDDIObjective:
    """
    EDDI objective.
    """

    _vamp_prior_info_gain_path = "vamp_prior_info_gain.json"

    # TODO #14636: Abstract init away from PVAE-only kwargs
    def __init__(self, model: IModelForObjective, sample_count: int, use_vamp_prior: bool = False, **kwargs):
        """
        Args:
            model (Model): Trained `Model` class to use.
            **config: Any kwargs required by model.get_information_gain such
                as sample_count (int) and use_vamp_prior (bool) for PVAE based
                models.
        """
        self._model = model
        self._sample_count = sample_count
        self._use_vamp_prior = use_vamp_prior
        self._batch_size = kwargs.get("batch_size", None)

    @classmethod
    def calc_and_save_vamp_prior_info_gain(cls, model, vamp_prior_data, sample_count):
        """
        Calculate information gain dictionary for VAMPPrior, and saves it to file.

        Args:
            model: Input data, shape (batch_size, input_dim)
            vamp_prior_data:
            sample_count: Number of samples to use in calculating information gain

        Returns:
            info_gain_dict: Dictionary containing estimated information gains for all features computed using VAMP
                Prior.
        """
        if issparse(vamp_prior_data[0]):
            # Need to convert to dense before converting to tensor.
            vamp_prior_data = tuple(mat.toarray() for mat in vamp_prior_data)
        empty_row = torch.zeros((1, model.variables.num_processed_cols), device=model.get_device())
        full_mask_row = torch.ones((1, model.variables.num_processed_cols), device=model.get_device())
        # Mark all features as observable (data_mask=1) and all features as currently unobserved (obs_mask=0).
        objective = cls(model, sample_count=sample_count)
        info_gain_dict = objective._information_gain(
            empty_row,
            full_mask_row,
            empty_row,
            as_array=False,
            vamp_prior_data=to_tensors(*vamp_prior_data, device=model.get_device()),
        )
        save_json(info_gain_dict, os.path.join(model.save_dir, cls._vamp_prior_info_gain_path))

    def get_information_gain(self, data: np.ndarray, data_mask: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
        """

        Calculate information gain for adding each observable group of features individually.

        Args:
            data (numpy array of shape (batch_size, feature_count)): Contains unprocessed, observed data.
            data_mask (numpy array of shape (batch_size, feature_count)): Contains mask where 1 is observed in the
                underlying data, 0 is missing.
            obs_mask (numpy array of shape (batch_size, feature_count)): Contains mask where 1 indicates a feature that has
                been observed before or during active learning, 0 a feature that has not yet been observed and could be
                queried (if the value also exists in the underlying dataset).

        Returns:
            rewards (numpy array of shape (batch_size, group_count)): Contains information gain for all observable groups of variables.
                Contains np.nan if the group of variables is observed already (1 in obs_mask for all of the group's variables)
                or not present in the data (0 in data_mask for all of the group's variables).
        """
        rewards = np.full((data.shape[0], len(self._model.variables.group_idxs)), np.nan)

        # Create mask indicating with a 1 which query groups cannot be observed in each row, because all of
        # group's features do not have a value in the underlying data (i.e. its value in data_mask is 0).
        group_masks = []
        for idxs in self._model.variables.group_idxs:
            group_masks.append((1 - data_mask[:, idxs].max(axis=1)).reshape(-1, 1))
        group_mask = np.hstack(group_masks)

        if self._use_vamp_prior:
            # For completely unobserved data, use precomputed info gain per variable
            vamp_rows = np.where(obs_mask.sum(axis=1) == 0)
            vamp_prior_info_dicts = read_json_as(
                os.path.join(self._model.save_dir, self._vamp_prior_info_gain_path), list
            )
            vamp_prior_info_array = np.array(pd.DataFrame(vamp_prior_info_dicts, index=[0]))[0]
            rewards[vamp_rows] = vamp_prior_info_array

            not_vamp_rows = np.nonzero(obs_mask.sum(axis=1))[0]
            data = data[not_vamp_rows]
            obs_mask = obs_mask[not_vamp_rows]
            data_mask = data_mask[not_vamp_rows]
        else:
            not_vamp_rows = np.arange(data.shape[0])

        if len(not_vamp_rows) > 0:
            batch_size = self._batch_size or len(not_vamp_rows)
            (
                proc_data,
                proc_data_mask,
                proc_obs_mask,
            ) = self._model.data_processor.process_data_and_masks(data, data_mask, obs_mask)
            dataloader = create_dataloader(
                not_vamp_rows,
                proc_data,
                proc_obs_mask,
                proc_data_mask,
                batch_size=batch_size,
                iterations=-1,
                sample_randomly=False,
                dtype=torch.float,
            )

            device = self._model.get_device()
            for rows, data_, obs_mask_, data_mask_ in dataloader:
                info_gains = self._information_gain(
                    data_.to(device),
                    data_mask_.to(device),
                    obs_mask_.to(device),
                    as_array=True,
                )
                rewards[rows.to(torch.long).cpu().numpy()] = info_gains

        # Remove estimates for unobservable (no values in observed data) groups of features
        rewards[group_mask.astype(bool)] = np.nan

        return rewards

    def _information_gain(
        self,
        data: torch.Tensor,
        data_mask: torch.Tensor,
        obs_mask: torch.Tensor,
        as_array: bool = False,
        vamp_prior_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Union[List[Dict[int, float]], np.ndarray]:
        """
        Calculate information gain of adding each observable group of features individually for a batch of data rows.

        Args:
            data (shape (batch_size, proc_feature_count)): processed, observed data.
            data_mask (shape (batch_size, proc_feature_count)): processed mask where 1 is observed in the
                underlying data, 0 is missing.
            obs_mask (shape (batch_size, proc_feature_count)): indicates which
                features have already been observed (i.e. which to condition the information gain calculations on).
            as_array (bool): When True will return info_gain values as an np.ndarray. When False (default) will return info
                gain values as a List of Dictionaries.
            vamp_prior_data: used to generate latent space samples when input data is completely unobserved
        Returns:
            rewards (List of Dictionaries or Numpy array): Length (batch_size) of dictionaries of the form {group_name : info_gain} where
                info_gain is np.nan if the group of variables is observed already (1 in obs_mask).
                If as_array is True, rewards is an array of shape (batch_size, group_count) instead.
        """
        assert obs_mask.shape == data.shape
        self._model.set_evaluation_mode()
        batch_size, feature_count = data.shape[0], self._model.variables.num_processed_non_aux_cols
        is_variable_to_observe = self._model.variables.get_variables_to_observe(data_mask)

        mask = data_mask * obs_mask

        # Repeat each row sample_count times to allow batch computation over all samples
        repeated_data = torch.repeat_interleave(data, self._sample_count, dim=0)
        repeated_mask = torch.repeat_interleave(mask, self._sample_count, dim=0)

        with torch.no_grad():  # Turn off gradient tracking for performance and to prevent numpy issues
            imputed = self._model.impute_processed_batch(
                data, mask, sample_count=self._sample_count, vamp_prior_data=vamp_prior_data, preserve_data=True
            )

            # Shape (sample_count, batch_size, feature_count)
            imputed = imputed.permute(1, 0, 2).reshape(self._sample_count * batch_size, feature_count)
            # Shape (sample_count * batch_size, feature_count) with neighbouring rows being different samples of the same imputation
            phi_idxs = self._model.variables.target_var_idxs

            # Compute q(z | x_o)
            q_o = self._model.encode(repeated_data, repeated_mask)

            if len(phi_idxs) > 0:
                # Compute q(z | x_o, x_phi)
                mask_o_phi = add_to_mask(self._model.variables, repeated_mask, phi_idxs)
                x_o_phi = add_to_data(self._model.variables, repeated_data, imputed, phi_idxs)
                q_o_phi = self._model.encode(x_o_phi, mask_o_phi)

            rewards_list = []
            for group_idxs in self._model.variables.group_idxs:
                # Can probably be further optimised by stacking masks for different vars
                if all(not is_variable_to_observe[idx] for idx in group_idxs):
                    diff = torch.full((batch_size,), np.nan)
                else:
                    mask_i_o = add_to_mask(self._model.variables, repeated_mask, group_idxs)
                    x_i_o = add_to_data(self._model.variables, repeated_data, imputed, group_idxs)
                    q_i_o = self._model.encode(x_i_o, mask_i_o)

                    kl1 = kl_divergence(q_i_o, q_o)  # Shape (sample_count * batch_size)
                    if len(phi_idxs) > 0:
                        mask_i_o_phi = add_to_mask(self._model.variables, mask_i_o, phi_idxs)
                        x_i_o_phi = add_to_data(self._model.variables, x_i_o, imputed, phi_idxs)
                        q_i_o_phi = self._model.encode(x_i_o_phi, mask_i_o_phi)

                        kl2 = kl_divergence(q_i_o_phi, q_o_phi)  # Shape (sample_count * batch_size)

                    else:
                        kl2 = torch.zeros_like(kl1)

                    diffs = (kl1 - kl2).cpu().numpy()
                    diffs = np.reshape(diffs, (self._sample_count, batch_size), order="F")
                    diff = np.mean(diffs, axis=0)  # Shape (batch_size)
                rewards_list.append(diff)

            rewards = np.vstack(rewards_list).T  # Shape (batch_size, feature_count)

            # Remove reward estimates for already observed groups of features
            # Also, note that the rewards are removed for unobservable (no values in observed data) groups of
            # features in the parent method (i.e. get_information_gain)
            rewards = self._remove_rewards_for_observed_groups(obs_mask, rewards)

            if not as_array:
                return [{idx: float(val) for idx, val in enumerate(row)} for row in rewards]
        return rewards

    def _remove_rewards_for_observed_groups(self, obs_mask: torch.Tensor, rewards: np.ndarray):
        # Remove reward estimates for already observed groups of features
        # This mask is for features, but rewards are for groups - collapse by collecting groups together.
        # Assume group not queriable if none of features queriable. 1 indicates already observed
        # (not queriable) so take min within each group.
        feature_mask = self._model.data_processor.revert_mask(obs_mask.cpu().numpy())
        group_masks = []
        for idxs in self._model.variables.group_idxs:
            group_masks.append(feature_mask[:, idxs].min(axis=1).reshape(-1, 1))
        group_mask = np.hstack(group_masks)
        rewards[group_mask.astype(bool)] = np.nan
        return rewards
