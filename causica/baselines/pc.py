import numpy as np
from castle.algorithms import PC as PC_alg

from ..datasets.variables import Variables
from ..utils.causality_utils import cpdag2dags
from .castle_causal_learner import CastleCausalLearner


class PC(CastleCausalLearner):
    """
    Child class of CastleCausalLearner which specifies PC_alg should be used for discovery
    """

    def __init__(self, model_id: str, variables: Variables, save_dir: str, random_seed: int = 0):
        super().__init__(model_id, variables, save_dir, PC_alg(), random_seed=random_seed)

    def get_adj_matrix(
        self, do_round: bool = True, samples: int = 100, most_likely_graph: bool = False, squeeze: bool = False
    ):
        """
        Draws a series of DAG samples from markov equivalence class. If enough samples are specified, all the DAGs in the equivalence class will be returned.
        """
        graph_samples = cpdag2dags(self.causal_learner.causal_matrix.astype(np.float64), samples=samples)

        return graph_samples[0] if samples == 1 and squeeze else graph_samples

    @classmethod
    def name(cls) -> str:
        return "pc"
