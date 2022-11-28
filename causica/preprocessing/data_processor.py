import logging
import warnings
from typing import Iterable, List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import csr_matrix, issparse
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm

from ..datasets.dataset import CausalDataset, Dataset, SparseDataset, TemporalDataset
from ..datasets.intervention_data import InterventionData
from ..datasets.itext_embedding_model import ITextEmbeddingModel
from ..datasets.variables import Variables
from .transforms import IdentityTransform, UnitScaler

EPSILON = 1e-5
logger = logging.getLogger(__name__)

V = TypeVar("V", np.ndarray, torch.Tensor)


# pylint: disable=protected-access
class DataProcessor:
    def __init__(
        self,
        variables: Variables,
        unit_scale_continuous: bool = True,
        standardize_data_mean: bool = False,
        standardize_data_std: bool = False,
        text_embedder: Optional[ITextEmbeddingModel] = None,
    ):
        """
        Args:
            variables (Variables): Information about variables/features used
                by this model.
            unit_scale_continuous (bool): Scale continuous variables to the range of [0, 1].
            standardize_data_mean (bool): Standardize continuous variables to mean=0
            standardize_data_std (bool): Standardize continuous variables to std=1
            text_embedder (ITextEmbeddingModel): text embedding model for
                processing text variables. If text variables present, it defaults to
                SentenceTransformerModel
        """
        if unit_scale_continuous and (standardize_data_mean or standardize_data_std):
            raise ValueError("Cannot unit scale and standardize variables simultanously.")
        self._variables = variables

        # Call unprocessed columns unproc_cols, processed columns proc_cols
        unproc_cols_by_type = self._variables.unprocessed_cols_by_type
        proc_cols_by_type = self._variables.processed_cols_by_type

        def flatten(lists):
            # Flatten proc_cols for continuous and binary unproc_cols, since they will be of form [[1], [2], ...]
            return [i for sublist in lists for i in sublist]

        if "binary" in unproc_cols_by_type:
            self._bin_unproc_cols = unproc_cols_by_type["binary"]
            self._bin_proc_cols = flatten(proc_cols_by_type["binary"])

            # Save contiguous regions containig binary features to allow for more efficient processing via slicing
            self._bin_unproc_regions = self.split_contiguous_sublists(self._bin_unproc_cols)
            self._bin_proc_regions = self.split_contiguous_sublists(self._bin_proc_cols)
            assert len(self._bin_unproc_regions) == len(self._bin_proc_regions)
        else:
            self._bin_unproc_cols, self._bin_proc_cols = [], []

        if "continuous" in unproc_cols_by_type:
            self._cts_unproc_cols = unproc_cols_by_type["continuous"]
            self._cts_proc_cols = flatten(proc_cols_by_type["continuous"])

            # Save contiguous regions containing continuous features to allow for more efficient processing via slicing
            if all(x.overwrite_processed_dim is None for x in self._variables):
                self._cts_unproc_regions = self.split_contiguous_sublists(self._cts_unproc_cols)
                self._cts_proc_regions = self.split_contiguous_sublists(self._cts_proc_cols)
            else:
                # For VAEM, we can only take single variable as region
                # to allow for processing/reverting mask
                self._cts_unproc_regions = [[col_id] for col_id in unproc_cols_by_type["continuous"]]
                self._cts_proc_regions = proc_cols_by_type["continuous"]
            assert len(self._cts_unproc_regions) == len(self._cts_proc_regions)
            if unit_scale_continuous:
                self._cts_normalizers = [
                    UnitScaler(variables[i] for i in unproc_region) for unproc_region in self._cts_unproc_regions
                ]
            elif standardize_data_mean or standardize_data_std:
                self._cts_normalizers = [
                    StandardScaler(with_mean=standardize_data_mean, with_std=standardize_data_std)
                    for _ in self._cts_unproc_regions
                ]
            else:
                self._cts_normalizers = [IdentityTransform()] * len(self._cts_unproc_regions)
        else:
            self._cts_unproc_cols, self._cts_proc_cols, self._cts_normalizers = [], [], []

        if "categorical" in unproc_cols_by_type:
            self._cat_unproc_cols = unproc_cols_by_type["categorical"]
            self._cat_proc_cols = flatten(proc_cols_by_type["categorical"])
            self._cat_proc_cols_grouped = proc_cols_by_type["categorical"]

            def get_lower(idx):
                return self._variables[idx].lower

            def get_upper(idx):
                return self._variables[idx].upper

            var_categories = [
                np.arange(int(get_lower(var_idx)), int(get_upper(var_idx)) + 1) for var_idx in self._cat_unproc_cols
            ]
            self._one_hot_encoder = OneHotEncoder(categories=var_categories, sparse=False, handle_unknown="ignore")
            # Fit on dummy data due to an issue in sklearn where the encoder needs to be fitted to data even if the
            # categories are specified upon creation.
            self._one_hot_encoder.fit(np.array([categories[0] for categories in var_categories]).reshape(1, -1))
        else:
            self._cat_unproc_cols, self._cat_proc_cols = [], []

        if "text" in unproc_cols_by_type:
            if text_embedder is None:
                raise ValueError("Found text but no embedder was provided.")
            # TODO: add assert on whether hidden space's dimensions agree
            self._txt_unproc_cols = unproc_cols_by_type["text"]
            self._txt_proc_cols = flatten(proc_cols_by_type["text"])
            self._txt_proc_cols_grouped = proc_cols_by_type["text"]

            self._text_embedder = text_embedder
        else:
            self._txt_unproc_cols, self._txt_proc_cols = [], []

        self._num_processed_cols = sum(var.processed_dim for var in self._variables)

    def process_data_and_masks(
        self,
        data: csr_matrix,
        data_mask: csr_matrix,
        *extra_masks: csr_matrix,
        batch_size: int = 1000,
    ) -> Tuple[csr_matrix, ...]:
        """
        Process and validate data, data mask and optionally any number of additional masks. These masks will all be applied
        to the data when performing data range validation, in case of e.g. dummy zero data that is masked out by an
        additional obs_mask.

        Args:
            data: Unprocessed data array
            data_mask: Data indicating which values in `data` are observed. Can be any dtype provided all values are
                either 0 or 1.
            extra_masks: Additional masks to be processed, if any. Can be any dtype provided all values are either 0 or
                1.
            batch_size: Batch size used during data preprocessing for sparse matrices.
        Returns:
            processed_data: Data with categorical variables expanded to a one-hot encoding, and features normalised.
            processed_data_mask: Boolean mask with categorical variables expanded to a one-hot encoding.
            processed_extra_masks: Any additional boolean masks with categorical variables expanded to a one-hot
                encoding.
        """
        if not issparse(data):
            (
                proc_data,
                proc_data_mask,
                *proc_extra_masks,
            ) = self._process_and_check_dense(data, data_mask, *extra_masks)
        else:
            # Break sparse data into smaller batches and preprocess each as a dense array. Somewhat inefficient but
            # allows us to reuse our preprocessing functions and keeps memory usage manageable.
            proc_data_list: List[csr_matrix] = []
            proc_data_mask_list: List[csr_matrix] = []
            proc_extra_masks_lists: Tuple[List[csr_matrix], ...] = tuple([] for mask in extra_masks)
            num_rows = data.shape[0]
            for start_idx in tqdm(range(0, num_rows, batch_size), desc="Data preprocessing"):
                stop_idx = min(start_idx + batch_size, num_rows)
                data_batch = data[start_idx:stop_idx].toarray()
                data_mask_batch = data_mask[start_idx:stop_idx].toarray()
                extra_masks_batch = tuple(mask[start_idx:stop_idx].toarray() for mask in extra_masks)

                # TODO: we will currently lose sparsity for rescaled continuous data here, since 0 will be mapped to
                # another value. We could multiply by the mask to zero out unobserved data but we need to make sure this
                # doesn't have any unintended consequences for cases with more complex masking, e.g. active learning
                (
                    proc_data_batch,
                    proc_data_mask_batch,
                    *proc_extra_masks_batch,
                ) = self._process_and_check_dense(data_batch, data_mask_batch, *extra_masks_batch)
                proc_data_list.append(csr_matrix(proc_data_batch))
                proc_data_mask_list.append(csr_matrix(proc_data_mask_batch))
                for mask_list, mask in zip(proc_extra_masks_lists, proc_extra_masks_batch):
                    mask_list.append(csr_matrix(mask))

            proc_data = sparse.vstack(proc_data_list, format="csr")
            proc_data_mask = sparse.vstack(proc_data_mask_list, format="csr")
            proc_extra_masks = tuple(
                sparse.vstack(proc_mask_list, format="csr") for proc_mask_list in proc_extra_masks_lists
            )

        return (proc_data, proc_data_mask, *proc_extra_masks)

    def _process_and_check_dense(self, data: np.ndarray, data_mask: np.ndarray, *extra_masks: np.ndarray):
        """
        Check validity of dense data and masks and process them.
        """
        combined_mask = data_mask
        for mask in extra_masks:
            combined_mask = combined_mask * mask
        self.check_data(data, combined_mask)
        self.check_mask(data_mask)
        for mask in extra_masks:
            self.check_mask(mask)
        proc_data = self.process_data(data)
        proc_data_mask = self.process_mask(data_mask)
        proc_extra_masks = tuple(self.process_mask(mask) for mask in extra_masks)
        return (proc_data, proc_data_mask, *proc_extra_masks)

    def process_intervention_data(
        self, intervention_data: Union[InterventionData, Iterable[InterventionData]]
    ) -> List[InterventionData]:
        """Preprocesses data in the InterventionData format and returns a list of processed InterventionData objects.



        Args:
            intervention_data (Union[InterventionData, Iterable[InterventionData]]): InterventionData object or list of
                InterventionData objects to be processed.

        Returns:
            List[InterventionData]: List of processed InterventionData objects.
        """
        if isinstance(intervention_data, InterventionData):
            intervention_data = [intervention_data]

        proc_intervention = [
            InterventionData(
                i.intervention_idxs,
                self.process_data_subset_by_group(i.intervention_values, i.intervention_idxs),
                self.process_data(i.test_data),
                i.conditioning_idxs,
                self.process_data_subset_by_group(i.conditioning_values, i.conditioning_idxs),
                i.effect_idxs,
                self.process_data_subset_by_group(i.intervention_reference, i.intervention_idxs),
                self.process_data(i.reference_data) if i.reference_data is not None else None,
            )
            for i in intervention_data
        ]

        return proc_intervention

    def process_dataset(
        self, dataset: Union[Dataset, CausalDataset, TemporalDataset, SparseDataset]
    ) -> Union[Dataset, CausalDataset, TemporalDataset, SparseDataset]:
        train_data, train_mask = self.process_data_and_masks(*dataset.train_data_and_mask)
        val_data, _ = dataset.val_data_and_mask
        if val_data is not None:
            val_data, val_mask = self.process_data_and_masks(*dataset.val_data_and_mask)
        else:
            val_data, val_mask = None, None
        test_data, _ = dataset.test_data_and_mask
        if test_data is not None:
            test_data, test_mask = self.process_data_and_masks(*dataset.test_data_and_mask)
        else:
            test_data, test_mask = None, None

        if isinstance(dataset, TemporalDataset):

            if dataset._intervention_data is not None:
                proc_intervention = self.process_intervention_data(dataset._intervention_data)
            else:
                proc_intervention = None

            # process counterfactual data
            if dataset._counterfactual_data is not None:
                proc_counterfactual = self.process_intervention_data(dataset._counterfactual_data)
            else:
                proc_counterfactual = None

            return TemporalDataset(
                train_data=train_data,
                train_mask=train_mask,
                transition_matrix=dataset._transition_matrix,
                adjacency_data=dataset._adjacency_data,
                intervention_data=proc_intervention,
                counterfactual_data=proc_counterfactual,
                val_data=val_data,
                val_mask=val_mask,
                test_data=test_data,
                test_mask=test_mask,
                variables=dataset._variables,
                data_split=dataset._data_split,
                train_segmentation=dataset.train_segmentation,
                test_segmentation=dataset._test_segmentation,
                val_segmentation=dataset._val_segmentation,
            )
        elif isinstance(dataset, CausalDataset):
            if dataset._intervention_data is not None:
                proc_intervention = self.process_intervention_data(dataset._intervention_data)
            else:
                proc_intervention = None

            # process counterfactual data
            if dataset._counterfactual_data is not None:
                proc_counterfactual = self.process_intervention_data(dataset._counterfactual_data)
            else:
                proc_counterfactual = None
            return CausalDataset(
                train_data,
                train_mask,
                dataset._adjacency_data,
                dataset._subgraph_data,
                proc_intervention,
                proc_counterfactual,
                val_data,
                val_mask,
                test_data,
                test_mask,
                variables=dataset.variables,
                data_split=dataset.data_split,
            )
        elif isinstance(dataset, (SparseDataset, Dataset)):
            return type(dataset)(
                train_data,
                train_mask,
                val_data,
                val_mask,
                test_data,
                test_mask,
                variables=dataset.variables,
            )
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    def check_mask(self, mask: np.ndarray) -> None:
        """
        Check mask contains 1 and 0 only
        """
        if len(mask.shape) != 2 or mask.shape[1] != len(self._variables):
            raise ValueError(
                "Mask must be 2D with shape (row_count, feature_count + aux_count)."
                f"Mask has shape {mask.shape} and feature_count is {len(self._variables)}."
            )
        bool_mask = mask.astype(bool)

        if not np.array_equal(mask, bool_mask):
            raise ValueError("Mask must contain 1 and 0 only.")

    def check_data(self, data: np.ndarray, mask: np.ndarray) -> None:
        """
        Check that each column of the data is valid with respect to the given variable definition.
        Raise an error if a discrete variable (binary or categorical) is not an integer or not within the specified range.
        Make a warning if a continuous variable is not within the specified range.
        Note that only observed values are checked.

        Args:
            variables: Variables object for data
            data: Unprocessed data array with shape (num_rows, num_features)
            mask: Mask indicting observed variables with shape (num_rows, num_features). 1 is observed, 0 is un-observed.
        """
        lower = np.array([var.lower for var in self._variables])
        upper = np.array([var.upper for var in self._variables])

        # Continuous variables
        cts_idxs = self._variables.continuous_idxs
        if len(cts_idxs) > 0:
            self.check_continuous_data(
                data=data[:, cts_idxs],
                mask=mask[:, cts_idxs],
                lower=lower[cts_idxs],
                upper=upper[cts_idxs],
                epsilon=EPSILON,
            )

        # Discrete variables
        disc_idxs = self._variables.discrete_idxs
        if len(disc_idxs) > 0:
            self.check_discrete_data(
                data=data[:, disc_idxs],
                mask=mask[:, disc_idxs],
                lower=lower[disc_idxs],
                upper=upper[disc_idxs],
                epsilon=EPSILON,
            )

    def check_continuous_data(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        epsilon: float,
    ) -> None:
        """
        Check if values in each column of the given continuous data are in the specified range. Make a warning
        if there is at least one value outside of the specified range. Note that only observed values are checked.

        Args:
            data: Unprocessed data array with shape (num_rows, num_features)
            mask: Mask indicting observed variables with shape (num_rows, num_features). 1 is observed, 0 is un-observed.
            lower: Array of column lower bounds with shape (num_features,)
            upper: Array of column upper bounds with shape (num_features,)
            epsilon: How close to the specified range we require values to be
        """
        # type annotation to avoid mypy error
        lower_diff: np.ndarray = data - lower
        higher_diff: np.ndarray = data - upper
        too_low_cols = np.any(lower_diff * mask < -1 * epsilon, axis=0)
        too_high_cols = np.any(higher_diff * mask > epsilon, axis=0)

        too_low = np.any(too_low_cols)
        too_high = np.any(too_high_cols)

        if too_low:
            warnings.warn(
                f"Data too low for continous variables {np.where(too_low_cols)[0]}",
                UserWarning,
            )
        if too_high:
            warnings.warn(
                f"Data too high for continous variables {np.where(too_high_cols)[0]}",
                UserWarning,
            )

    def check_discrete_data(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        epsilon: float,
    ) -> None:
        """
        Check if values in each column of the given discrete (binary and categorical) data are in the specified range.
        Raise an error if there is at least one value outside of the specified range.
        Additionally, assert that all the given values are integers. Note that only observed values are checked.

        Args:
            data: Unprocessed data array with shape (num_rows, num_features)
            mask: Mask indicting observed variables with shape (num_rows, num_features). 1 is observed, 0 is un-observed.
            lower: Array of column lower bounds with shape (num_features,)
            upper: Array of column upper bounds with shape (num_features,)
            epsilon: How close to the specified range we require values to be
        """
        lower_diff: np.ndarray = data - lower
        higher_diff: np.ndarray = data - upper
        too_low_cols = np.any(lower_diff * mask < -1 * epsilon, axis=0)
        too_high_cols = np.any(higher_diff * mask > epsilon, axis=0)

        too_low = np.any(too_low_cols)
        too_high = np.any(too_high_cols)

        if too_low and too_high:
            raise ValueError(
                f"Data too low for discrete variables {np.where(too_low_cols)[0]} \n"
                f"Data too high for discrete variables {np.where(too_high_cols)[0]}"
            )
        if too_low:
            raise ValueError(f"Data too low for discrete variables {np.where(too_low_cols)[0]}")
        if too_high:
            raise ValueError(f"Data too high for discrete variables {np.where(too_high_cols)[0]}")

        # Check all unmasked values are integer-valued.
        observed_data: np.ndarray = data * mask
        is_integer = np.floor_divide(observed_data, 1) == observed_data
        assert np.all(is_integer)

    def process_data(self, data: np.ndarray) -> np.ndarray:
        """
        Returns the processed data and fits the normalizers the first time executed.

        Args:
            data: Array of shape (num_rows, feature_count + aux_count)
                or (num_timeseries, num_timesteps, feature_count + aux_count). If it's temporal data
                the data will be flattened (num_rows, feature_count + aux_count) and the columns
                will be processed irrespective of the timeseries.
        Returns:
            processed_data: Array of shape (num_rows, num_processed_cols) or (num_timeseries, num_timesteps, num_processed_cols)
        """

        is_temporal = len(data.shape) == 3

        if is_temporal:
            orig_shape = data.shape
            data = data.reshape((np.prod(orig_shape[:2]), -1))

        num_rows, _ = data.shape

        # If all features are binary, no processing required so short-circuit here
        if len(self._cts_unproc_cols) == 0 and len(self._cat_unproc_cols) == 0:
            return data.astype(float)

        processed_data = np.full((num_rows, self._num_processed_cols), fill_value=np.nan)

        # Iterate through each contiguous subarray of features of each type. Can guarantee that these regions will line
        # up between processed and unprocessed arrays since we don't change the feature order. We do this since
        # accessing/writing slices is much more efficient in NumPy than fancy indexing.
        # TODO: if we can sort/unsort features by type during processing without breaking anything, then we can simply
        # do one slice of the array per feature type and not need all this extra complexity.

        if self._bin_unproc_cols:
            for unproc_region, proc_region in zip(self._bin_unproc_regions, self._bin_proc_regions):
                unproc_start, unproc_end = unproc_region[0], unproc_region[-1] + 1
                proc_start, proc_end = proc_region[0], proc_region[-1] + 1
                processed_data[:, proc_start:proc_end] = data[:, unproc_start:unproc_end].astype(float)

        if self._cts_unproc_cols:
            for unproc_region, proc_region, normalizer in zip(
                self._cts_unproc_regions, self._cts_proc_regions, self._cts_normalizers
            ):
                unproc_start, unproc_end = unproc_region[0], unproc_region[-1] + 1
                proc_start, proc_end = proc_region[0], proc_region[-1] + 1
                cts_unproc_data = data[:, unproc_start:unproc_end].astype(float)

                # Make sure the transform has been fitted
                try:
                    check_is_fitted(normalizer)
                except NotFittedError:
                    normalizer.fit(cts_unproc_data)

                processed_data[:, proc_start:proc_end] = normalizer.transform(cts_unproc_data)

        if self._cat_unproc_cols:
            # Don't currently split into separate contiguous subarrays for categorical vars since we only want a single
            # one-hot encoder for simplicity.
            cat_unproc_data = data[:, self._cat_unproc_cols].astype(float)
            processed_data[:, self._cat_proc_cols] = self._one_hot_encoder.transform(cat_unproc_data)

        if self._txt_unproc_cols:
            processed_data[:, self._txt_proc_cols] = self._text_embedder.encode(data[:, self._txt_unproc_cols])

        if is_temporal:
            processed_data = processed_data.reshape(list(orig_shape[:-1]) + [-1])

        return processed_data

    def process_data_subset_by_group(
        self, data: Optional[np.ndarray], idxs: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Args:
            data: Array of shape (num_rows, num_unprocessed_cols_subset) or (num_unprocessed_cols_subset)
                or (num_timeseries, num_rows, num_unprocessed_cols_subset).
                Data should be ordered by group, and then by variables within that group in the same
                order as the main dataset. If the data is temporal it will be flattened to
                (num_rows, num_unprocessed_cols_subset) and the columnswill be processed irrespective of the timeseries.
            idxs: Array indicating the ordered indices of the groups represented in the data.
        Returns:
            processed_data: Array of shape (num_rows, num_processed_cols_subset) or (num_processed_cols_subset)
                or (num_timeseries, num_rows, num_processed_cols_subset)
        """
        # Add statement idxs is None, to avoid mypy error: None type has no __iter__. I assume if data is None or idxs is None, just return None.
        if data is None or idxs is None:  # Helpful when calling from `process_dataset`
            return None

        is_temporal = len(data.shape) == 3

        # For temporal data, remove time index from idxs.
        if idxs.ndim > 1:
            idxs = idxs[..., 0]

        if is_temporal:
            orig_shape = data.shape
            data = data.reshape((np.prod(orig_shape[:2]), -1))

        if len(data.shape) == 0:
            data = np.array([data.item()])
            num_rows = 1
        elif len(data.shape) == 1:
            num_rows = 1
        else:
            num_rows, _ = data.shape
        pseudodata = np.zeros((num_rows, self._variables.num_processed_cols))

        start = 0
        for i in idxs:
            for j in self._variables.group_idxs[i]:
                unproc_dim = self._variables[j].unprocessed_dim
                pseudodata[:, self._variables.unprocessed_cols[j]] = data[..., start : (start + unproc_dim)]
                start += unproc_dim

        processed_pseudodata = self.process_data(pseudodata)

        output_num_cols = self._variables.group_mask[idxs, :].sum()
        return_data = np.full((num_rows, output_num_cols), fill_value=np.nan)

        start = 0
        for i in idxs:
            for j in self._variables.group_idxs[i]:
                proc_dim = self._variables[j].processed_dim
                return_data[:, start : (start + proc_dim)] = processed_pseudodata[:, self._variables.processed_cols[j]]
                start += proc_dim

        if len(data.shape) == 1:
            return_data = return_data.squeeze(0)

        if is_temporal:
            return_data = return_data.reshape(list(orig_shape[:-1]) + [-1])

        return return_data

    def process_mask(self, mask: V) -> V:
        """
        Args:
            mask: Array/Tensor of shape (num_rows, feature_count + aux_count) taking values 0 or 1
        Returns:
            processed_mask: Boolean array of shape (num_rows, num_processed_cols)
        """
        num_rows, _ = mask.shape

        if isinstance(mask, np.ndarray):  # If numpy array opperate on bools
            processed_mask = np.zeros((num_rows, self._num_processed_cols), dtype=bool)
        elif isinstance(mask, torch.Tensor):  # If torch tensors operate on floats
            processed_mask = torch.zeros(
                (num_rows, self._num_processed_cols),
                dtype=mask.dtype,
                device=mask.device,
            )
        else:
            raise ValueError("Wrong type of mask object")

        if self._bin_unproc_cols:
            for unproc_region, proc_region in zip(self._bin_unproc_regions, self._bin_proc_regions):
                unproc_start, unproc_end = unproc_region[0], unproc_region[-1] + 1
                proc_start, proc_end = proc_region[0], proc_region[-1] + 1
                processed_mask[:, proc_start:proc_end] = mask[:, unproc_start:unproc_end]

        if self._cts_unproc_cols:
            for unproc_region, proc_region in zip(self._cts_unproc_regions, self._cts_proc_regions):
                unproc_start, unproc_end = unproc_region[0], unproc_region[-1] + 1
                proc_start, proc_end = proc_region[0], proc_region[-1] + 1
                processed_mask[:, proc_start:proc_end] = mask[:, unproc_start:unproc_end]

        if self._cat_unproc_cols:
            for var, proc_cols in zip(self._cat_unproc_cols, self._cat_proc_cols_grouped):
                # Index with var:var+1 to return 2D array rather than 1D to allow broadcasting.
                processed_mask[:, proc_cols] = mask[:, var : var + 1]

        if self._txt_unproc_cols:
            for var, proc_cols in zip(self._txt_unproc_cols, self._txt_proc_cols_grouped):
                # Index with var:var+1 to return 2D array rather than 1D to allow broadcasting.
                processed_mask[:, proc_cols] = mask[:, var : var + 1]

        return processed_mask

    def revert_mask(self, mask: V) -> V:
        """
        Revert processed mask into unprocessed form (i.e. squash categorical/text var indices).

        Args:
            variables:
            mask: Numpy array/Torch tensor with shape (num_rows, input_count)

        Returns:
            data: Numpy array/Torch tensor with shape (num_rows, feature_count + aux_count)
        """
        proc_cols_to_delete = []
        for idx, var in enumerate(self._variables):
            if var.type_ not in {"categorical", "text"} and var.overwrite_processed_dim is not None:
                continue
            cols = self._variables.processed_cols[idx]
            # Delete all columns except for first one
            proc_cols_to_delete += cols[1:]
        proc_cols_to_stay = [col for col in range(mask.shape[1]) if col not in proc_cols_to_delete]
        return mask[:, proc_cols_to_stay]

    def revert_data(self, data: np.ndarray) -> np.ndarray:
        """
        Undo processing to return output in the same form as the input. Sort-of-inverse of process_data.
        This involves reversing the squash operation for continuous variables, changing one-hot
        categorical variables into a single natural number and reordering data.

        Args:
            data: Numpy array with shape (num_rows, input_count)

        Returns:
            data: Numpy array with shape (num_rows, feature_count + aux_count)
        """
        # revert_data() is only called on imputed data, which is inherently dense, so we assume a sparse matrix is never
        # passed into this method.

        num_rows, _ = data.shape

        unprocessed_data = np.empty((num_rows, self._variables.num_unprocessed_cols), dtype=object)

        if self._bin_unproc_cols:
            for unproc_region, proc_region in zip(self._bin_unproc_regions, self._bin_proc_regions):
                unproc_start, unproc_end = unproc_region[0], unproc_region[-1] + 1
                proc_start, proc_end = proc_region[0], proc_region[-1] + 1
                unprocessed_data[:, unproc_start:unproc_end] = data[:, proc_start:proc_end]

        if self._cts_unproc_cols:
            for unproc_region, proc_region, normalizer in zip(
                self._cts_unproc_regions, self._cts_proc_regions, self._cts_normalizers
            ):
                unproc_start, unproc_end = unproc_region[0], unproc_region[-1] + 1
                proc_start, proc_end = proc_region[0], proc_region[-1] + 1
                unprocessed_data[:, unproc_start:unproc_end] = normalizer.inverse_transform(
                    data[:, proc_start:proc_end]
                )

        if self._cat_unproc_cols:
            unprocessed_data[:, self._cat_unproc_cols] = self._one_hot_encoder.inverse_transform(
                data[:, self._cat_proc_cols]
            )

        if self._txt_unproc_cols:
            unprocessed_data[:, self._txt_unproc_cols] = self._text_embedder.decode(data[:, self._txt_proc_cols])

        return unprocessed_data

    @staticmethod
    def split_contiguous_sublists(ints: List[int]) -> List[List[int]]:
        """
        Map from list of ints to list of contiguous sublists. E.g. [1,2,4,6,7] -> [[1,2],[4],[6,7]]. Assumes input list
        is sorted.
        """
        out: List[List[int]] = []
        for i in ints:
            if len(out) == 0:
                out.append([i])
            elif i == out[-1][-1] + 1:
                out[-1].append(i)
            else:
                out.append([i])
        return out
