from castle.algorithms import ICALiNGAM as ICALiNGAM_alg

from ..datasets.variables import Variables
from .castle_causal_learner import CastleCausalLearner


class ICALiNGAM(CastleCausalLearner):
    def __init__(self, model_id: str, variables: Variables, save_dir: str, random_seed: int = 0):
        super().__init__(model_id, variables, save_dir, ICALiNGAM_alg(), random_seed=random_seed)

    @classmethod
    def name(cls) -> str:
        return "icalingam"
