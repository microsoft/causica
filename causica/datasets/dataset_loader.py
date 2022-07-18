import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..datasets.dataset import Dataset, SparseDataset
from ..datasets.variables import Variables
from ..utils.io_utils import read_json_as

logger = logging.getLogger(__name__)


class DatasetLoader(ABC):
    """
    Abstract dataset loader class.

    To instantiate this class, these functions need to be implemented:
        split_data_and_load_dataset
        load_predefined_dataset
    """

    _variables_file = "variables.json"
    _negative_sampling_file = "negative_sampling_levels.csv"
    _predefined_data_split = {"train_idxs": "predefined", "val_idxs": "predefined", "test_idxs": "predefined"}

    def __init__(self, dataset_dir: str):
        """
        Args:
            dataset_dir: Directory in which the dataset files are contained, or will be saved if not present.
        """
        self.dataset_dir = dataset_dir

    @abstractmethod
    def split_data_and_load_dataset(
        self,
        test_frac: float,
        val_frac: float,
        random_state: Union[int, Tuple[int, int]],
        max_num_rows: Optional[int] = None,
        **kwargs,
    ) -> Union[Dataset, SparseDataset]:
        """
        Load the data from disk and make the train/val/test split to instantiate a dataset.
        The data is split deterministically given the random state. If the given random state is a pair of integers,
        the first is used to extract test set and the second is used to extract the validation set from the remaining data.
        If only a single integer is given as random state it is used for both.
        Args:
            test_frac: Fraction of data to put in the test set.
            val_frac: Fraction of data to put in the validation set.
            random_state: An integer or a tuple of integers to be used as the splitting random state.
            max_num_rows: Maximum number of rows to include when reading data files.
            negative_sample: Whether to perform negative sampling after loading the dataset. Negative sampling requires
                a file negative_sampling_levels.csv in the dataset folder, and negative samples for each row will be
                drawn from features of a greater level than all those observed in the row.
        Returns:
            dataset: Dataset or SparseDataset object, holding the data and variable metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def load_predefined_dataset(self, max_num_rows: Optional[int] = None, **kwargs) -> Union[Dataset, SparseDataset]:
        """
        Load the data from disk and use the predefined train/val/test split to instantiate a dataset.
        Args:
            max_num_rows: Maximum number of rows to include when reading data files.
        Returns:
            dataset: Dataset or SparseDataset object, holding the data and variable metadata.
        """
        raise NotImplementedError

    def _load_variables_dict(self) -> Optional[Dict[str, List[Any]]]:
        """
        Load variables info dictionary from a file if it exists.
        Args:
            dataset_dir: Directory in which the dataset files are contained.
        Returns:
            variables_dict: If not None, dictionary containing metadata for each variable (column) in the input data.
        """
        variables_path = os.path.join(self.dataset_dir, self._variables_file)
        logger.info(f"Variables info {variables_path} exists: {os.path.exists(variables_path)}.")
        variables_dict = read_json_as(variables_path, dict) if os.path.exists(variables_path) else None
        return variables_dict

    @staticmethod
    def load_negative_sampling_levels(negative_sampling_levels_path: str, variables: Variables) -> Dict[int, int]:
        path_exists = os.path.exists(negative_sampling_levels_path)
        logger.info(f"Negative sampling levels at {negative_sampling_levels_path} exist: {path_exists}.")
        if path_exists:
            negative_sampling_df = pd.read_csv(negative_sampling_levels_path, names=["col_id", "level"])
            # Map dataset's feature ids to our corresponding variable index in [0, ..., num_vars - 1]
            negative_sampling_df["col_id"] = negative_sampling_df["col_id"].map(variables.col_id_to_var_index)
            return DatasetLoader._convert_negative_sampling_df_to_dict(negative_sampling_df)
        else:
            raise FileNotFoundError(f"Negative sampling levels file not found: {negative_sampling_levels_path}.")

    @staticmethod
    def _convert_negative_sampling_df_to_dict(df: pd.DataFrame) -> Dict[int, int]:
        """
        Convert DataFrame with columns "col_id" and "level" to dictionary mapping col_id -> level.
        """
        transposed_df = df.set_index("col_id").T
        # to_dict() sets dict keys as lists - in this case we just have 1 element so index 0th element.
        levels = {i: j[0] for i, j in transposed_df.to_dict("list").items()}
        return levels

    # There is a bug adding types to data/data_mask here as there is an open issue with overloading static methods in
    # MyPy, see https://github.com/python/mypy/issues/7781.
    @staticmethod
    def negative_sample(data, data_mask, levels: Dict[int, int]):
        """
        Apply negative sampling to an input data array and mask. Each feature is assigned an integer level in the
        dictionary `levels`, corresponding to e.g. question difficulty, and for each row we sample additional negative
        (0) values from all features with a level greater the max level of all features observed in the row. Negative
        samples will be drawn from these features until an equal number of positive (>0) and negative (0) values exist
        in the row, if possible.

        Note: this method currently assumes that negative responses in the dataset take a value of 0.

        Args:
            data: Binary data array
            data_mask: Boolean data mask
            levels: A dictionary mapping from variable id -> level.

        Returns:
            data: Input array with added negative samples
            data_mask: Input mask with added observations for negative samples
        """
        assert data.ndim == 2
        num_rows, _ = data.shape
        all_col_idxs = set(range(data.shape[1]))
        # Make copies to avoid modifying data/mask in place.
        data = data.copy()
        data_mask = data_mask.copy()
        # LIL format matrices are much more efficient than CSR if we are adding elements and thus changing the sparsity
        # structure.
        if issparse(data):
            data = data.tolil()
            data_mask = data_mask.tolil()
        for row_idx in tqdm(range(num_rows), desc="Negative sampling"):
            # Indexing i:i+1 as a slight hack to prevent row dimension being dropped for dense data row.
            data_row = data[row_idx : row_idx + 1, :]
            data_mask_row = data_mask[row_idx : row_idx + 1, :]

            row_max_level = max(levels[i] for i in data_mask_row.nonzero()[1])
            observed_col_idxs = set(data_mask_row.nonzero()[1])
            unobserved_col_idxs = all_col_idxs - observed_col_idxs
            negative_sampling_candidates = [i for i in unobserved_col_idxs if levels[i] > row_max_level]

            if negative_sampling_candidates:
                # Do enough negative samples that num_positive = num_negative, if enough candidates are available.
                num_to_sample = (data_row[data_mask_row.nonzero()] > 0).sum() - (
                    data_row[data_mask_row.nonzero()] == 0
                ).sum()
                # Can't sample more than the total number of candidates available.
                num_to_sample = min(num_to_sample, len(negative_sampling_candidates))
                if num_to_sample > 0:
                    choices = np.random.choice(negative_sampling_candidates, size=num_to_sample, replace=False)
                    data[row_idx, choices] = 0
                    data_mask[row_idx, choices] = 1
        if issparse(data):
            data = data.tocsr()
            data_mask = data_mask.tocsr()
        return data, data_mask

    def _apply_negative_sampling(self, variables, train_data, train_mask, val_data, val_mask, test_data, test_mask):
        negative_sampling_levels_path = os.path.join(self.dataset_dir, self._negative_sampling_file)
        negative_sampling_levels = self.load_negative_sampling_levels(negative_sampling_levels_path, variables)

        train_data, train_mask = self.negative_sample(train_data, train_mask, negative_sampling_levels)
        if test_data is not None and test_mask is not None:
            test_data, test_mask = self.negative_sample(test_data, test_mask, negative_sampling_levels)
        if val_data is not None:
            val_data, val_mask = self.negative_sample(val_data, val_mask, negative_sampling_levels)
        else:
            val_data, val_mask = None, None

        return train_data, train_mask, val_data, val_mask, test_data, test_mask

    @classmethod
    def _generate_data_split(
        cls,
        rows: List[int],
        test_frac: float,
        val_frac: float,
        random_state: Union[int, Tuple[int, int], List[int]] = (0, 0),
    ) -> Tuple[List[int], List[int], List[int], Dict[str, List[int]]]:
        """
        Split the given list of row indices into three lists using the given test and validation fraction.
        This function is deterministic given the random state. If the given random state is a pair of integers,
        the first is used to extract test set rows and the second is used to extract the validation set rows from the remaining rows.
        If only a single integer is given as random state it is used for both.
        Args:
            rows: List of row indices to be split.
            test_frac: Fraction of rows to put in the test set.
            val_frac: Fraction of rows to put in the validation set.
            random_state: An integer or a pair of integers to be used as the splitting random state.
        Returns:
            train_rows: List of row indices to assigned to the train set.
            val_rows: List of row indices to assigned to the validation set.
            test_rows: List of row indices to assigned to the test set.
            data_split: Dictionary record about how the row indices were split.
        """
        if isinstance(random_state, (list, tuple)):
            random_state = cast(Tuple[int, int], random_state)
            test_random_state = random_state[0]
            val_random_state = random_state[1]
        else:
            random_state = cast(int, random_state)
            test_random_state = random_state
            val_random_state = random_state

        cls._validate_val_frac_test_frac(test_frac, val_frac)

        if test_frac > 0:
            train_val_rows, test_rows = train_test_split(rows, test_size=test_frac, random_state=test_random_state)
        else:
            train_val_rows = rows
            test_rows = []

        if val_frac > 0:
            val_frac = val_frac / (1 - test_frac)
            train_rows, val_rows = train_test_split(train_val_rows, test_size=val_frac, random_state=val_random_state)
        else:
            train_rows = train_val_rows
            val_rows = []

        train_rows.sort()
        val_rows.sort()
        test_rows.sort()
        data_split = {
            "train_idxs": [int(id) for id in train_rows],
            "val_idxs": [int(id) for id in val_rows],
            "test_idxs": [int(id) for id in test_rows],
        }

        return train_rows, val_rows, test_rows, data_split

    @classmethod
    def _validate_val_frac_test_frac(cls, test_frac: float, val_frac: float) -> None:
        """
        Assert that the provided values for the test and validation fractions are valid.
        Args:
            test_frac: Test fraction.
            val_frac: Validation fraction.
        """
        assert test_frac is not None and val_frac is not None
        assert 0 <= test_frac < 1
        assert 0 <= val_frac < 1
        assert test_frac + val_frac < 1
