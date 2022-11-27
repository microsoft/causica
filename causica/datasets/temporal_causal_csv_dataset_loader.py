import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .causal_csv_dataset_loader import CausalCSVDatasetLoader
from .csv_dataset_loader import CSVDatasetLoader
from .dataset import TemporalDataset
from .variables import Variables

logger = logging.getLogger(__name__)


class TemporalCausalCSVDatasetLoader(CausalCSVDatasetLoader):
    """
    Load a dataset from a CSV file in tabular format, i.e. where each row is an individual datapoint and each
    column (excluding timeseries_column_index) is a feature. For multiple time-series, all series are concatenated together,
    and we have one column to specify the series number. The shape of the data will be [tot_length, (D+1)], where tot_length is
    the time-series length after concatenation and D is the number of variables.
    Load an adjacency matrix from a CSV file contained in the same data directory.
    Load a variable number of intervention vectors together with their corresponding intervened data
    from CSV files contained within the same data directory.
    """

    _adjacency_data_file = "adj_matrix.npy"
    _transition_matrix_file = "transition_matrix.npy"

    def split_data_and_load_dataset(  # type:ignore
        self,
        test_frac: float,
        val_frac: float,
        random_state: Union[int, Tuple[int, int]] = 0,
        max_num_rows: Optional[int] = None,
        negative_sample: bool = False,
        timeseries_column_index: Optional[int] = None,
        **kwargs,
    ) -> TemporalDataset:
        """
        Load the data from memory and make the train/val/test split to instantiate a dataset.
        The data is split deterministically along the time axis.

        Args:
            test_frac: Fraction of data to put in the test set.
            val_frac: Fraction of data to put in the validation set.
            max_num_rows: Maximum number of rows to include when reading data files.
            negative_sample: Whether to perform negative sampling after loading the dataset. Negative sampling requires
                a file negative_sampling_levels.csv in the dataset folder, and negative samples for each row will be
                drawn from features of a greater level than all those observed in the row.
            timeseries_column_index: Indicate which column specifies the time-series index

        Returns:
            temporal_dataset: TemporalDataset object, holding the data and variable metadata as well as
            the transition matrix as a np.ndarray, the adjacency matrix as a np.ndarray and a list of InterventionData
            objects, each containing an intervention vector and samples.
        """
        assert not negative_sample, "current time series split does not support negative samples"
        assert timeseries_column_index is not None, "For temporal data loader, column must be specified"

        logger.info("Create temporal dataset.")

        logger.info(
            f"Splitting temporal data to load the dataset: test fraction: {test_frac}, validation fraction: {val_frac}."
        )
        data_path = os.path.join(self.dataset_dir, self._all_data_file)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"The required temporal data file not found: {data_path}.")

        data, mask = self.read_csv_from_file(data_path, max_num_rows=max_num_rows)
        train_rows, val_rows, test_rows = self.temporal_train_test_val_split(
            data[:, timeseries_column_index],
            test_frac,
            val_frac,
            random_state=random_state,
        )

        train_data, test_data = data[train_rows, :], data[test_rows, :]
        train_mask, test_mask = mask[train_rows, :], mask[test_rows, :]
        val_data = data[val_rows, :] if len(val_rows) > 0 else None
        val_mask = mask[val_rows, :] if len(val_rows) > 0 else None

        # data split
        data_split = {
            "train_idxs": [int(id) for id in train_rows],
            "val_idxs": [int(id) for id in val_rows] if len(val_rows) > 0 else None,
            "test_idxs": [int(id) for id in test_rows],
        }

        # variables
        variables_dict = self._load_variables_dict()
        variables = Variables.create_from_data_and_dict(train_data, train_mask, variables_dict)

        adjacency_data = self._get_adjacency_data()
        intervention_data = self._load_data_from_intervention_files(max_num_rows)
        counterfactual_data = self._load_data_from_intervention_files(max_num_rows, True)
        transition_matrix = self._get_transition_matrix()

        temporal_dataset = TemporalDataset(
            train_data=train_data,
            train_mask=train_mask,
            transition_matrix=transition_matrix,
            adjacency_data=adjacency_data,
            intervention_data=intervention_data,
            counterfactual_data=counterfactual_data,
            val_data=val_data,
            val_mask=val_mask,
            test_data=test_data,
            test_mask=test_mask,
            variables=variables,
            data_split=data_split,
            train_segmentation=None,
            test_segmentation=None,
            val_segmentation=None,
        )

        temporal_dataset = self.process_dataset(temporal_dataset, timeseries_column_index)
        return temporal_dataset

    def temporal_train_test_val_split(
        self,
        series_column: np.ndarray,
        test_frac: float,
        val_frac: float,
        random_state: Union[int, Tuple[int, int]],
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        This method will output the row selections for training, test and validation data. It currently supports
        splitting single and multiple time series. For single time series, it will split according to time index.
        For multiple time series, it will split according to series number. For example, if we have 10 time series, and
        the test fraction is 0.2. Then, we will select 2 out of 10 series to be the test series. The underlying assumption
        here is that we will not encounter extremely unbalanced data in real life.
        Args:
            series_column: the column that contains the time series number. E.g. [0,0,...,1,1,...,2,...]
            test_frac: test fraction for data splitting
            val_frac: val data fraction.
            random_state: random number for data splitting.
        """
        # Process random seed
        if random_state is None:
            test_random_state = 0
            val_random_state = 0
        elif isinstance(random_state, tuple):
            test_random_state, val_random_state = random_state[0], random_state[1]
        else:
            test_random_state, val_random_state = random_state, random_state
        # Generate unique time-series index and also their corresponding length.
        series_index = np.unique(series_column, return_counts=False)
        if len(series_index) == 1:
            # Only has one time-series index, then we split the data according to time-index.
            rows = list(np.arange(series_column.shape[0]))
            train_rows, val_rows, test_rows, _ = self._split_single_series(rows, test_frac, val_frac)
            return train_rows, val_rows, test_rows
        else:
            # Multiple time series, we split the data as described in the docs.
            num_series = len(series_index)
            num_test = int(test_frac * num_series)
            num_val = int(val_frac * num_series)
            num_train = num_series - num_test - num_val
            assert num_train > 0, "Not enough data split for training."
            assert num_train < num_series, "No remaining data for test and val"
            # shuffle the list and select training
            np.random.RandomState(test_random_state).shuffle(series_index)
            train_idx_selection = series_index[:num_train]
            series_index = series_index[num_train:]
            # shuffle the list and select test and val
            np.random.RandomState(val_random_state).shuffle(series_index)
            test_idx_selection = series_index[:num_test]
            if len(test_idx_selection) == len(series_index):
                logger.warning("There is no data for val")
                val_idx_selection = []
            else:
                val_idx_selection = series_index[num_test:]

            # Generate row index based on selected time series
            # Select the data according to the time-series index.
            train_rows = []
            for train_series_index in train_idx_selection:
                # This maps the time-series index to a list of rows index. In the above example, time-series index 0 will map to a list [0,1,2,...,96].
                # So if we pick series [0,2] for training, then the train_rows will be [0,1,2,..,96,98]
                train_rows += self._map_from_series_to_rows(train_series_index, series_column)

            test_rows = []
            for test_series_index in test_idx_selection:
                test_rows += self._map_from_series_to_rows(test_series_index, series_column)

            val_rows = []
            if len(val_idx_selection) > 0:
                for val_series_index in val_idx_selection:
                    val_rows += self._map_from_series_to_rows(val_series_index, series_column)
            return train_rows, val_rows, test_rows

    def _map_from_series_to_rows(self, series_index: Union[int, np.int_], series_column: np.ndarray) -> List[int]:
        """
        This will map the target time-series index to a list of index specifying the locations of that time series.
        E.g. data containing 4 time-series with index [0,1,2,3] with length [97,1,1,1]. Then series_column will be [0,0,...,1,2,3].
        If series_index is 0, then it will return [0,1,2,...,96]. If series_index is 2, it will return [98].
        Args:
            series_index: target time series number
            series_column: the array that contains the series numbers.
        """
        index_list = np.argwhere(series_column == series_index).squeeze()
        if len(index_list.shape) == 0:
            index_list_ = [index_list.tolist()]
        else:
            index_list_ = index_list.tolist()
        return index_list_

    def load_predefined_dataset(
        self,
        max_num_rows: Optional[int] = None,
        negative_sample: bool = False,
        **kwargs,
    ) -> TemporalDataset:
        """
        Load the data from memory and use the predefined train/val/test split to instantiate a dataset.

        Args:
            max_num_rows: Maximum number of rows to include when reading data files.
            negative_sample: Whether to perform negative sampling after loading the dataset. Negative sampling requires
                a file negative_sampling_levels.csv in the dataset folder, and negative samples for each row will be
                drawn from features of a greater level than all those observed in the row.

        Returns:
            temporal_dataset: TemporalDataset object, holding the data and variable metadata as well as
            the transition matrix as a np.ndarray, the adjacency matrix as a np.ndarray and a list of InterventionData
            objects, each containing an intervention vector and samples.
        """
        column_index = kwargs.get("column_index", 0)
        dataset = CSVDatasetLoader.load_predefined_dataset(self, max_num_rows, negative_sample)

        logger.info("Create temporal dataset.")

        adjacency_data = self._get_adjacency_data()
        intervention_data = self._load_data_from_intervention_files(max_num_rows)
        counterfactual_data = self._load_data_from_intervention_files(max_num_rows, True)
        transition_matrix = self._get_transition_matrix()
        temporal_dataset = dataset.to_temporal(
            adjacency_data,
            intervention_data,
            transition_matrix,
            counterfactual_data,
        )
        temporal_dataset = self.process_dataset(temporal_dataset, column_index)
        return temporal_dataset

    def process_dataset(self, dataset: TemporalDataset, timeseries_column_index: int) -> TemporalDataset:
        """
        This is to process the temporal dataset so that the data does not contain time-series index column (the first column)
        and generate corresponding segmentation list. Then update the TemporalDataset for train/test/val data and segmentation index.

        Args:
            dataset: Temporal dataset with temporal format.
            timeseries_column_index: index indicate which column specifies the time series index.
        """
        # pylint: disable=protected-access
        train_data, train_mask = dataset._train_data, dataset._train_mask
        test_data, test_mask = dataset._test_data, dataset._test_mask
        val_data, val_mask = dataset._val_data, dataset._val_mask

        # Process variable_dict by removing the index variable
        variable_dict = self._load_variables_dict()
        if variable_dict is not None:
            assert "variables" in variable_dict, "variable dict must define variables."
            assert (
                len(variable_dict["variables"]) == train_data.shape[-1]
            ), f"Variables dims ({len(variable_dict['variables'])}) must match the data dims ({train_data.shape[-1]})"
            # Delete column index variable from variable_dict
            del variable_dict["variables"][timeseries_column_index]

        # Process dataset
        proc_train_data, proc_train_mask, train_segmentation = self._remove_series_index_and_generate_segmentation(
            train_data, train_mask, timeseries_column_index
        )
        if test_data is not None and test_mask is not None:
            proc_test_data, proc_test_mask, test_segmentation = self._remove_series_index_and_generate_segmentation(
                test_data, test_mask, timeseries_column_index
            )
        if val_data is not None and val_mask is not None:
            proc_val_data, proc_val_mask, val_segmentation = self._remove_series_index_and_generate_segmentation(
                val_data, val_mask, timeseries_column_index
            )

        # Re-infer the variables through the processed variable dict
        variables = Variables.create_from_data_and_dict(proc_train_data, proc_train_mask, variables_dict=variable_dict)

        # Modify the dataset
        dataset._train_data, dataset._train_mask, dataset.train_segmentation = (
            proc_train_data,
            proc_train_mask,
            train_segmentation,
        )
        dataset._variables = variables
        if test_data is not None:
            dataset._test_data, dataset._test_mask, dataset._test_segmentation = (
                proc_test_data,
                proc_test_mask,
                test_segmentation,
            )
        if val_data is not None:
            dataset._val_data, dataset._val_mask, dataset._val_segmentation = (
                proc_val_data,
                proc_val_mask,
                val_segmentation,
            )

        return dataset

    def _remove_series_index_and_generate_segmentation(
        self, data: np.ndarray, mask: np.ndarray, timeseries_column_index: int
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """
        This removes the column of the data where we store the time-series index.
        It also generates the corresponding index segmentations list.
        NOTE: This assumes that the series indices are contiguous.
        Args:
            data: Temporal data.
            mask: the corresponding masks of the data
            timeseries_column_index: the column index specifying the time series number
        """
        num_series = len(np.unique(data[:, timeseries_column_index]))
        start_idx = 0
        series_seg = []
        for _ in range(num_series):
            cur_series_number = data[start_idx, timeseries_column_index]
            series_range = np.argwhere(data[:, timeseries_column_index] == cur_series_number)
            series_seg.append((int(series_range[0]), int(series_range[-1])))
            start_idx = int(series_range[-1]) + 1
        proc_data = np.delete(data, timeseries_column_index, 1)
        proc_mask = np.delete(mask, timeseries_column_index, 1)
        return proc_data, proc_mask, series_seg

    @classmethod
    def _split_single_series(
        cls, rows: List[int], test_frac: float, val_frac: float, *_args, **_kwargs
    ) -> Tuple[List[int], List[int], List[int], Dict[str, List[int]]]:
        """
        Split the given list of row indices into three lists using the given test and validation fraction.
        The data is split deterministically along the time axis.

        Args:
            rows: List of row indices to be split.
            test_frac: Fraction of rows to put in the test set.
            val_frac: Fraction of rows to put in the validation set.
        Returns:
            train_rows: List of row indices to assigned to the train set.
            val_rows: List of row indices to assigned to the validation set.
            test_rows: List of row indices to assigned to the test set.
            data_split: Dictionary record about how the row indices were split.
        """
        cls._validate_val_frac_test_frac(test_frac, val_frac)

        rows.sort()
        num_samples = len(rows)

        num_test = int(num_samples * test_frac)

        val_frac = val_frac / (1 - test_frac)
        num_val = int((num_samples - num_test) * val_frac)

        if num_test > 0:
            test_rows = rows[-num_test:]
        else:
            test_rows = []

        if num_val > 0:
            val_rows = rows[:-num_test][-num_val:]
        else:
            val_rows = []

        train_rows = rows[: -(num_test + num_val)]

        train_rows.sort()
        val_rows.sort()
        test_rows.sort()
        data_split = {
            "train_idxs": [int(id) for id in train_rows],
            "val_idxs": [int(id) for id in val_rows],
            "test_idxs": [int(id) for id in test_rows],
        }

        return train_rows, val_rows, test_rows, data_split

    def _get_adjacency_data(self):

        adjacency_data_path = os.path.join(self.dataset_dir, self._adjacency_data_file)

        adjacency_file_exists = all([os.path.exists(adjacency_data_path)])

        if not adjacency_file_exists:
            logger.info("DAG adjacency matrix not found.")
            adjacency_data = None
        else:
            logger.info("DAG adjacency matrix found.")
            adjacency_data = np.load(adjacency_data_path)

        return adjacency_data

    def _get_transition_matrix(self):

        transition_matrix_path = os.path.join(self.dataset_dir, self._transition_matrix_file)

        transition_matrix_file_exists = all([os.path.exists(transition_matrix_path)])

        if not transition_matrix_file_exists:
            logger.info("Transition matrix not found.")
            transition_matrix = None
        else:
            logger.info("Transition matrix found.")
            transition_matrix = np.load(transition_matrix_path)

        return transition_matrix
