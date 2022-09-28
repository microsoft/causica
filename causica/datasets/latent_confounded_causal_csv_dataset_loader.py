import logging
import os
from typing import Optional, Tuple, Union

from .causal_csv_dataset_loader import CausalCSVDatasetLoader
from .dataset import LatentConfoundedCausalDataset

logger = logging.getLogger(__name__)


class LatentConfoundedCausalCSVDatasetLoader(CausalCSVDatasetLoader):
    """Child class of CausalCSVDatasetLoader that also loads directed and bidirected adjacency matrices to represented
    observed variables that are confounded by a latent variable."""

    _directed_adjacency_data_file = "directed_adjacency_matrix.csv"
    _bidirected_adjacency_data_file = "bidirected_adjacency_matrix.csv"

    def split_data_and_load_dataset(
        self,
        test_frac: float,
        val_frac: float,
        random_state: Union[int, Tuple[int, int]],
        max_num_rows: Optional[int] = None,
        negative_sample: bool = False,
        **kwargs,
    ) -> LatentConfoundedCausalDataset:
        """Load the data from disk and make the train/val/test split to instantiate a dataset.

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
            confounded_causal_dataset: LatentConfoundedCausalDataset object, holding the data and variable metadata as well as
                the adjacency matrices as a np.ndarray and a list of InterventionData objects, each containing an
                intervention vector and samples.
        """

        dataset = super(CausalCSVDatasetLoader, self).split_data_and_load_dataset(
            test_frac, val_frac, random_state, max_num_rows, negative_sample
        )

        logger.info("Create confounded causal dataset.")

        directed_adjacency_data, known_directed_subgraph_mask = self._get_directed_adjacency_data()
        bidirected_adjacency_data, known_bidirected_subgraph_mask = self._get_bidirected_adjacency_data()
        intervention_data = self._load_data_from_intervention_files(max_num_rows)
        counterfactual_data = self._load_data_from_intervention_files(max_num_rows, True)
        confounded_causal_dataset = dataset.to_latent_confounded_causal(
            directed_adjacency_data,
            known_directed_subgraph_mask,
            bidirected_adjacency_data,
            known_bidirected_subgraph_mask,
            intervention_data,
            counterfactual_data,
        )
        return confounded_causal_dataset

    def load_predefined_dataset(
        self,
        max_num_rows: Optional[int] = None,
        negative_sample: bool = False,
        **kwargs,
    ) -> LatentConfoundedCausalDataset:
        """Load the data from disk and use the predefined train/val/test split to instantiate a dataset.

        Args:
            max_num_rows: Maximum number of rows to include when reading data files.
            negative_sample: Whether to perform negative sampling after loading the dataset. Negative sampling requires
                a file negative_sampling_levels.csv in the dataset folder, and negative samples for each row will be
                drawn from features of a greater level than all those observed in the row.

        Returns:
            confounded_causal_dataset: LatentConfoundedCausalDataset object, holding the data and variable metadata as well as
            the adjacency matrix as a np.ndarray and a list of InterventionData objects, each containing an intervention
            vector and samples.
        """
        dataset = super(CausalCSVDatasetLoader, self).load_predefined_dataset(max_num_rows, negative_sample)

        logger.info("Create confounded causal dataset.")

        directed_adjacency_data, known_directed_subgraph_mask = self._get_directed_adjacency_data()
        bidirected_adjacency_data, known_bidirected_subgraph_mask = self._get_bidirected_adjacency_data()
        intervention_data = self._load_data_from_intervention_files(max_num_rows)
        counterfactual_data = self._load_data_from_intervention_files(max_num_rows, True)
        confounded_causal_dataset = dataset.to_latent_confounded_causal(
            directed_adjacency_data,
            known_directed_subgraph_mask,
            bidirected_adjacency_data,
            known_bidirected_subgraph_mask,
            intervention_data,
            counterfactual_data,
        )
        return confounded_causal_dataset

    def _get_directed_adjacency_data(self):

        directed_adjacency_data_path = os.path.join(self.dataset_dir, self._directed_adjacency_data_file)
        directed_adjacency_file_exists = os.path.exists(directed_adjacency_data_path)

        if not directed_adjacency_file_exists:
            logger.info("Directed adjacency matrix not found.")
            directed_adjacency_data, mask = None, None
        else:
            logger.info("Directed adjacency matrix found.")
            directed_adjacency_data, mask = self.read_csv_from_file(directed_adjacency_data_path)

        return directed_adjacency_data, mask

    def _get_bidirected_adjacency_data(self):

        bidirected_adjacency_data_path = os.path.join(self.dataset_dir, self._bidirected_adjacency_data_file)
        bidirected_adjacency_file_exists = os.path.exists(bidirected_adjacency_data_path)

        if not bidirected_adjacency_file_exists:
            logger.info("Bidireted adjacency matrix not found.")
            bidirected_adjacency_data, mask = None, None
        else:
            logger.info("Bidirected adjacency matrix found.")
            bidirected_adjacency_data, mask = self.read_csv_from_file(bidirected_adjacency_data_path)

        return bidirected_adjacency_data, mask
