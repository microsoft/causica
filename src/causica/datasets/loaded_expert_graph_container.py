import fsspec
import numpy as np
import torch

from causica.distributions.adjacency import ExpertGraphContainer
from causica.fsspec_helpers import get_storage_options_for_path


class LoadedExpertGraphContainer(ExpertGraphContainer):
    """Loads an expert graph container from a file path."""

    def __init__(self, graph_path: str, confidence: float, scale: float):
        """
        Args:
            graph_path: fsspec compatible path to a saved graph, with np.nan indicating undecided values.
                This can be saved as a .npy file or a .csv file.
            confidence: See ExpertGraphContainer
            scale: See ExpertGraphContainer
        """
        storage_options = get_storage_options_for_path(graph_path)
        with fsspec.open(graph_path, **storage_options) as f:
            if graph_path.endswith(".csv"):
                prior_adj_matrix = np.loadtxt(f, delimiter=",")
            else:
                prior_adj_matrix = np.load(f)
        prior_mask = ~np.isnan(prior_adj_matrix)
        prior_adj_matrix = np.nan_to_num(prior_adj_matrix)
        super().__init__(
            dag=torch.tensor(prior_adj_matrix), mask=torch.tensor(prior_mask), confidence=confidence, scale=scale
        )
