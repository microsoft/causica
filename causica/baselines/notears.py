from castle.algorithms import Notears
from castle.algorithms import NotearsMLP as NotearsMLP_alg
from castle.algorithms import NotearsSob as NotearsSob_alg

from ..datasets.variables import Variables
from .castle_causal_learner import CastleCausalLearner


class NotearsLinear(CastleCausalLearner):
    def __init__(self, model_id: str, variables: Variables, save_dir: str, random_seed: int = 0):
        super().__init__(model_id, variables, save_dir, Notears(), random_seed=random_seed)

    @classmethod
    def name(cls) -> str:
        return "notears_linear"


class NotearsMLP(CastleCausalLearner):
    def __init__(self, model_id: str, variables: Variables, save_dir: str, random_seed: int = 0):
        super().__init__(model_id, variables, save_dir, NotearsMLP_alg(), random_seed=random_seed)

    @classmethod
    def name(cls) -> str:
        return "notears_mlp"


class NotearsSob(CastleCausalLearner):
    def __init__(self, model_id: str, variables: Variables, save_dir: str, random_seed: int = 0):
        super().__init__(model_id, variables, save_dir, NotearsSob_alg(), random_seed=random_seed)

    @classmethod
    def name(cls) -> str:
        return "notears_sob"
