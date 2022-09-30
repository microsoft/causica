from typing import Callable, Optional, Tuple

import numpy as np
import torch
from castle.common import BaseLearner
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import fisherz

from ..datasets.variables import Variables
from ..utils.causality_utils import admg2dag, pag2admgs
from .castle_causal_learner import CastleCausalLearner


class CastleFCI(BaseLearner):
    """Child class of BaseLearner which uses causallearn's FCI implementation."""

    def __init__(self, ci_test: Callable = fisherz):
        super().__init__()

        self.ci_test = ci_test

    def learn(self, data):
        self._causal_matrix = fci(data, self.ci_test)[0].graph


class FCI(CastleCausalLearner):
    """Child class of CastleCausalLearner which specifies that the FCI algorithm should be used for causal discovery."""

    def __init__(self, model_id: str, variables: Variables, save_dir: str, random_seed: int = 0):
        super().__init__(model_id, variables, save_dir, CastleFCI(), random_seed=random_seed)

    def get_adj_matrix(
        self, do_round: bool = True, samples: int = 100, most_likely_graph: bool = False, squeeze: bool = False
    ):
        """Draws DAG samples from the Markov equivalence class returned by FCI. If enough samples are specified, then
        all the DAGs in the MEC are returned.

        Args:
            do_round: All samples are rounded, so not needed.. Defaults to True.
            samples: Number of samples to draw at random from the MEC. Defaults to 100.
            squeeze: Whether to squeeze the batch dimension if a single sample is specified. Defaults to False.

        Returns:
            Samples from the MEC discovered by the FCI algorithm.
        """
        directed_adjs, bidirected_adjs = self.get_admg_matrices(samples)
        adjs = np.stack(
            [
                admg2dag(torch.as_tensor(directed_adj), torch.as_tensor(bidirected_adj)).numpy()
                for directed_adj, bidirected_adj in zip(directed_adjs, bidirected_adjs)
            ]
        )

        return adjs[0] if samples == 1 and squeeze else adjs

    def get_admg_matrices(self, samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Samples from the variational distribution over ADMGs.

        Args:
            do_round: Whether to round the edge probabilities. Defaults to True.
            samples: Number of samples to draw. Defaults to 100.
            most_likely_graph: Whether to sample the most likely graph. Defaults to False.

        Raises:
            NotImplementedError: If adjacency mode is not found.

        Returns:
            The directed and bidirected adjacency matrices.
        """

        return pag2admgs(self.causal_learner.causal_matrix, samples=samples)

    @classmethod
    def name(cls) -> str:
        return "fci"
