import logging
from typing import Optional, Tuple, Union

import pandas as pd

from ..datasets.csv_dataset_loader import CSVDatasetLoader
from ..datasets.dataset import Dataset

logger = logging.getLogger(__name__)


class PandasDatasetLoader(CSVDatasetLoader):
    """
    Load a dataset from a Pandas object
    """

    def split_data_and_load_dataset(  # type: ignore
            self,
            pandas_data: pd.DataFrame,
            test_frac: float,
            val_frac: float,
            random_state: Union[int, Tuple[int, int]],
            max_num_rows: Optional[int] = None,
            negative_sample: bool = False,
            **kwargs,
    ) -> Dataset:
        """
        Load the data from memory and make the train/val/test split to instantiate a dataset.
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
            dataset: Dataset object, holding the data and variable metadata.
        """
        logger.info(
            f"Splitting data to load the dataset: test fraction: {test_frac}, validation fraction: {val_frac}."
        )

        if max_num_rows is not None:
            pandas_data = pandas_data.iloc[:max_num_rows]

        data, mask = self.process_data(pandas_data.to_numpy())
        return self._make_dataset(
            data, mask, negative_sample, test_frac, val_frac, random_state
        )

    def load_predefined_dataset(
            self,
            max_num_rows: Optional[int] = None,
            negative_sample: bool = False,
            **kwargs,
    ) -> Dataset:
        raise NotImplementedError()
