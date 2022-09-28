import logging
import os
from typing import List, Optional, Tuple, TypedDict, Union

import numpy as np

from ..utils.io_utils import read_json_as
from .csv_dataset_loader import CSVDatasetLoader
from .dataset import CausalDataset
from .intervention_data import InterventionData, InterventionDataContainer


class OptionalInterventionDataDict(TypedDict, total=False):
    reference: np.ndarray
    effect_idx: np.ndarray
    reference_samples: np.ndarray


class InterventionDataDict(OptionalInterventionDataDict):
    conditioning: np.ndarray
    intervention: np.ndarray
    intervention_samples: np.ndarray


logger = logging.getLogger(__name__)


class CausalCSVDatasetLoader(CSVDatasetLoader):
    """
    Load a dataset from a CSV file in tabular format, i.e. where each row is an individual datapoint and each
    column is a feature. Load an adjacency matrix from a CSV file contained in the same data directory.
    Load a variable number of intervention vectors together with their corresponding intervened data
    from CSV files contained within the same data directory.
    """

    _intervention_data_basefilename = "interventions"
    _counterfactual_data_basefilename = "counterfactuals"
    _adjacency_data_file = "adj_matrix.csv"

    def split_data_and_load_dataset(
        self,
        test_frac: float,
        val_frac: float,
        random_state: Union[int, Tuple[int, int]],
        max_num_rows: Optional[int] = None,
        negative_sample: bool = False,
        **kwargs,
    ) -> CausalDataset:
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
            causal_dataset: CausalDataset object, holding the data and variable metadata as well as
            the adjacency matrix as a np.ndarray and a list of InterventionData objects, each containing an intervention
            vector and samples.
        """

        dataset = super().split_data_and_load_dataset(test_frac, val_frac, random_state, max_num_rows, negative_sample)

        logger.info("Create causal dataset.")

        adjacency_data, known_subgraph_mask = self._get_adjacency_data()
        intervention_data = self._load_data_from_intervention_files(max_num_rows)
        counterfactual_data = self._load_data_from_intervention_files(max_num_rows, True)
        causal_dataset = dataset.to_causal(adjacency_data, known_subgraph_mask, intervention_data, counterfactual_data)
        return causal_dataset

    def load_predefined_dataset(
        self,
        max_num_rows: Optional[int] = None,
        negative_sample: bool = False,
        **kwargs,
    ) -> CausalDataset:
        """
        Load the data from disk and use the predefined train/val/test split to instantiate a dataset.

        Args:
            max_num_rows: Maximum number of rows to include when reading data files.
            negative_sample: Whether to perform negative sampling after loading the dataset. Negative sampling requires
                a file negative_sampling_levels.csv in the dataset folder, and negative samples for each row will be
                drawn from features of a greater level than all those observed in the row.

        Returns:
            causal_dataset: CausalDataset object, holding the data and variable metadata as well as
            the adjacency matrix as a np.ndarray and a list of InterventionData objects, each containing an intervention
            vector and samples.
        """
        dataset = super().load_predefined_dataset(max_num_rows, negative_sample)

        logger.info("Create causal dataset.")

        adjacency_data, known_subgraph_mask = self._get_adjacency_data()
        intervention_data = self._load_data_from_intervention_files(max_num_rows)
        counterfactual_data = self._load_data_from_intervention_files(max_num_rows, True)
        causal_dataset = dataset.to_causal(adjacency_data, known_subgraph_mask, intervention_data, counterfactual_data)
        return causal_dataset

    def _get_adjacency_data(self):

        adjacency_data_path = os.path.join(self.dataset_dir, self._adjacency_data_file)

        adjacency_file_exists = all([os.path.exists(adjacency_data_path)])

        if not adjacency_file_exists:
            logger.info("DAG adjacency matrix not found.")
            adjacency_data = None
            mask = None
        else:
            logger.info("DAG adjacency matrix found.")
            adjacency_data, mask = self.read_csv_from_file(adjacency_data_path)

        return adjacency_data, mask

    def _load_data_from_intervention_files(
        self, max_num_rows: Optional[int], is_counterfactual: bool = False
    ) -> Optional[List[InterventionData]]:
        """Loads data from files following the InterventionData format.

        This is used for loading interventional data as well as counterfactual data.

        Args:
            max_num_rows (Optional[int]): Maximum number of rows to include when reading data files.
            is_counterfactual (bool): Whether to load counterfactual data.

        Returns:
            Optional[List[InterventionData]]: List of InterventionData objects.
        """

        intervention_data_path = os.path.join(
            self.dataset_dir,
            self._counterfactual_data_basefilename if is_counterfactual else self._intervention_data_basefilename,
        )

        if os.path.exists(intervention_data_path + ".csv"):
            logger.info("Intervention data csv found.")
            raw_intervention_csv_data, mask = self.read_csv_from_file(
                intervention_data_path + ".csv", max_num_rows=max_num_rows
            )
            intervention_data = self._parse_csv_intervention_data(
                raw_intervention_csv_data, mask, is_counterfactual=is_counterfactual
            )
        elif os.path.exists(intervention_data_path + ".npy"):
            logger.info("Intervention data npy found.")

            raw_intervention_npy_data = np.load(intervention_data_path + ".npy", allow_pickle=True).item()
            assert isinstance(raw_intervention_npy_data, dict)  # mypy

            intervention_data_container = InterventionDataContainer.from_dict(raw_intervention_npy_data)
            intervention_data_container.validate(counterfactual=is_counterfactual)
            intervention_data = intervention_data_container.environments

        elif os.path.exists(intervention_data_path + ".json"):
            logger.info("Intervention data json found.")
            raw_intervention_json_data = read_json_as(intervention_data_path + ".json", dict)
            intervention_data_container = InterventionDataContainer.from_dict(raw_intervention_json_data)
            intervention_data_container.validate(counterfactual=is_counterfactual)
            intervention_data = intervention_data_container.environments
        else:
            logger.info("Intervention data not found.")
            intervention_data = None

        return intervention_data

    @classmethod
    def _parse_csv_intervention_data(
        cls, raw_intervention_data: np.ndarray, mask: np.ndarray, is_counterfactual: bool = False
    ) -> List[InterventionData]:
        """
        TODO: re-structure this method into smaller sub-methods to increase readability

           Parse the raw data from the interventions.csv file, separating the intervened variables, their intervened values and samples from the intervened distribution.
           Also, if they exist, retrieves indinces of effect variables, reference variables, data generated with reference interventions, conditioning indices and conditioning variables.
           If they do not exist, those fields of the InterventionData object are populated with None.
           Expects format of interventions.csv to be 5xN_vars columns. The order is [conditioning_cols, intervention_cols, reference_cols, effect_mask_cols, data_cols].
           It is infered automatically which rows correspond to the same intervention.

            Args:
                raw_intervention_data: np.ndarray read directly from interventions.csv
                mask: Corresponding mask, where observed values are 1 and np.nan values (representing non-intervened variables) are 0.
                is_counterfactual: Whether the data is counterfactual or not.

            Returns:
                causal_dataset: List of instances of InterventionData, one per each intervention.

        """

        Ncols = int(raw_intervention_data.shape[1] / 5)
        Nrows = raw_intervention_data.shape[0]

        # Split into rows that contain conditioning vectors, intervention vectors, referrence vectors, effect vectors, and rows that contain samples
        conditioning_cols = raw_intervention_data[:, :Ncols].astype(float)
        conditioning_mask_cols = mask[:, :Ncols]

        intervention_cols = raw_intervention_data[:, Ncols : 2 * Ncols].astype(float)
        intervention_mask_cols = mask[:, Ncols : 2 * Ncols]

        reference_cols = raw_intervention_data[:, 2 * Ncols : 3 * Ncols].astype(float)
        reference_mask_cols = mask[:, 2 * Ncols : 3 * Ncols]

        effect_mask_cols = mask[:, 3 * Ncols : 4 * Ncols]

        sample_cols = raw_intervention_data[:, -Ncols:].astype(float)

        # Iterate over file rows, checking if they contain the start of a new intervention
        intervention_data = []
        intervention_start_row = 0
        has_ref = False

        # Process first row

        # identify conditionioning variable indices and their values
        conditioning_idxs = np.where(conditioning_mask_cols[0, :] == 1)[0]
        conditioning_values = conditioning_cols[0, conditioning_idxs]

        if is_counterfactual:
            assert np.all(conditioning_mask_cols[0, :] == 1), "Counterfactual data expects the conditioning to be full."
            cf_conditioning_idxs = [conditioning_idxs]
            cf_conditioning_values = [conditioning_values]

        # identify intervention variable indices and their values
        intervention_idxs = np.where(intervention_mask_cols[0, :] == 1)[0]
        intervention_values = intervention_cols[0, intervention_idxs]

        # identify reference variable indices and their values
        reference_idxs = np.where(reference_mask_cols[0, :] == 1)[0]
        reference_values = reference_cols[0, reference_idxs]
        assert len(reference_idxs) == 0, "reference identified in data without previous intervention"

        # identify effect variable indices and their values
        effect_idxs = np.where(effect_mask_cols[0, :] == 1)[0]

        # Process all remaining rows
        for n_row in range(1, Nrows):

            next_conditioning_idxs = np.where(conditioning_mask_cols[n_row, :] == 1)[0]
            next_conditioning_values = conditioning_cols[n_row, next_conditioning_idxs]

            next_intervention_idxs = np.where(intervention_mask_cols[n_row, :] == 1)[0]
            next_intervention_values = intervention_cols[n_row, next_intervention_idxs]

            next_reference_idxs = np.where(reference_mask_cols[n_row, :] == 1)[0]
            next_reference_values = reference_cols[n_row, next_reference_idxs]

            next_effect_idxs = np.where(effect_mask_cols[n_row, :] == 1)[0]

            intervention_change = list(next_intervention_idxs) != list(intervention_idxs) or list(
                next_intervention_values
            ) != list(intervention_values)
            # check whether we're handling counterfactual data or not
            if not is_counterfactual:
                intervention_change = intervention_change or (
                    list(next_conditioning_idxs) != list(conditioning_idxs)
                    or list(next_conditioning_values) != list(conditioning_values)
                )

            ref_start = len(reference_idxs) == 0 and len(next_reference_idxs) > 0

            # check for the start of reference data for an intervention
            if ref_start:
                assert not has_ref, "there must be no more than one reference dataset per intervention dataset"
                assert (
                    n_row > intervention_start_row
                ), "there must be interevention test data for there to be reference data"
                has_ref = True
                intervention_end_row = n_row

                reference_idxs = next_reference_idxs
                reference_values = next_reference_values

            # decide data for a given intervention has finished based on where the intervened indices or values change
            if intervention_change:
                # Ensure that we dont intervene, condition or measure effect on same variable
                assert not set(intervention_idxs) & set(conditioning_idxs) & set(effect_idxs)
                # Ensure that reference incides are empty or match the treatment indices
                assert reference_idxs == intervention_idxs or len(reference_idxs) == 0

                # Check for references, conditioning and effects. Set to None if they are not present in the data
                if has_ref:
                    reference_data = sample_cols[intervention_end_row:n_row]
                    has_ref = False
                else:
                    intervention_end_row = n_row
                    reference_data = None
                    reference_values = None

                if is_counterfactual:
                    conditioning_idxs = np.stack(cf_conditioning_idxs)
                    conditioning_values = np.stack(cf_conditioning_values)

                    assert np.all(
                        conditioning_mask_cols[n_row, :] == 1
                    ), "Counterfactual data expects the conditioning to be full."
                    cf_conditioning_idxs = [next_conditioning_idxs]
                    cf_conditioning_values = [next_conditioning_values]

                intervention_data.append(
                    InterventionData(
                        conditioning_idxs=None if len(conditioning_idxs) == 0 else conditioning_idxs,
                        conditioning_values=None if len(conditioning_values) == 0 else conditioning_values,
                        effect_idxs=None if len(effect_idxs) == 0 else effect_idxs,
                        intervention_idxs=intervention_idxs,
                        intervention_values=intervention_values,
                        intervention_reference=reference_values,
                        test_data=sample_cols[intervention_start_row:intervention_end_row],
                        reference_data=reference_data,
                    )
                )

                intervention_start_row = n_row

                intervention_idxs = next_intervention_idxs
                intervention_values = next_intervention_values

                conditioning_idxs = next_conditioning_idxs
                conditioning_values = next_conditioning_values

                effect_idxs = next_effect_idxs

                reference_idxs = next_reference_idxs
                reference_values = next_reference_values

            elif is_counterfactual:
                assert np.all(
                    conditioning_mask_cols[n_row, :] == 1
                ), "Counterfactual data expects the conditioning to be full."
                cf_conditioning_idxs += [next_conditioning_idxs]
                cf_conditioning_values += [next_conditioning_values]

        # Process final intervention
        if has_ref:
            reference_data = sample_cols[intervention_end_row:]
        else:
            intervention_end_row = n_row + 1
            reference_data = None
            reference_values = None

        if is_counterfactual:
            conditioning_idxs = np.stack(cf_conditioning_idxs)
            conditioning_values = np.stack(cf_conditioning_values)

        intervention_data.append(
            InterventionData(
                conditioning_idxs=None if len(conditioning_idxs) == 0 else conditioning_idxs,
                conditioning_values=None if len(conditioning_values) == 0 else conditioning_values,
                effect_idxs=None if len(effect_idxs) == 0 else effect_idxs,
                intervention_idxs=intervention_idxs,
                intervention_values=intervention_values,
                intervention_reference=reference_values,
                test_data=sample_cols[intervention_start_row:intervention_end_row],
                reference_data=reference_data,
            )
        )
        return intervention_data
