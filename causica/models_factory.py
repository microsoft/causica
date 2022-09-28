import os
from typing import Any, Dict, Type, Union
from uuid import uuid4

from .baselines.do_why import DoWhy
from .baselines.dynotears import Dynotears
from .baselines.end2end_causal.deci_dowhy import DECIDoWhy
from .baselines.end2end_causal.informed_deci import InformedDECI
from .baselines.end2end_causal.pc_dowhy import PCDoWhy
from .baselines.end2end_causal.pc_informed_deci import PCInformedDECI
from .baselines.end2end_causal.true_graph_dowhy import TrueGraphDoWhy
from .baselines.grandag import GraNDAG
from .baselines.icalingam import ICALiNGAM
from .baselines.notears import NotearsLinear, NotearsMLP, NotearsSob
from .baselines.pc import PC
from .baselines.pcmci_plus import PCMCI_Plus
from .baselines.varlingam import VARLiNGAM
from .datasets.variables import Variables
from .models.deci.deci import DECI
from .models.deci.deci_gaussian import DECIGaussian
from .models.deci.deci_spline import DECISpline
from .models.deci.fold_time_deci import FoldTimeDECI
from .models.imodel import IModel
from .models.point_net import PointNet, SparsePointNet
from .models.set_encoder_base_model import SetEncoderBaseModel
from .models.transformer_set_encoder import TransformerSetEncoder
from .models.visl import VISL

MODEL_SUBCLASSES: Dict[str, Type[IModel]] = {
    model.name(): model  # type: ignore
    for model in (
        # Models
        DECI,
        VISL,
        DECIGaussian,
        DECISpline,
        FoldTimeDECI,
        # Baselines
        DoWhy,
        GraNDAG,
        ICALiNGAM,
        InformedDECI,
        NotearsLinear,
        NotearsLinear,
        NotearsMLP,
        NotearsSob,
        DECIDoWhy,
        PC,
        PCDoWhy,
        PCInformedDECI,
        TrueGraphDoWhy,
        VARLiNGAM,
        Dynotears,
        PCMCI_Plus,
    )
}


class ModelClassNotFound(NotImplementedError):
    pass


def create_model(
    model_name: str,
    models_dir: str,
    variables: Variables,
    device: Union[str, int],
    model_config_dict: Dict[str, Any],
    model_id: str = None,
) -> IModel:
    """
    Get an instance of an implementation of the `Model` class.

    Args:
        model_name (str): String corresponding to concrete instance of `Model` class.
        models_dir (str): Directory to save model information in.
        variables (Variables): Information about variables/features used
                by this model.
        model_config_dict (dictionary): Any other parameters needed by a specific concrete class. Of
            the form {arg_name: arg_value}. e.g. {"sample_count": 10}
        device (str or int): Name of Torch device to create the model on. Valid options are 'cpu', 'gpu', or a device ID
            (e.g. 0 or 1 on a two-GPU machine).
        model_id (str): String specifying GUID for model. A GUID will be generated if not provided.

    Returns:
        Instance of concrete implementation of `Model` class.
    """
    # Create anything needed for all model types.
    model_id = model_id if model_id is not None else str(uuid4())
    save_dir = os.path.join(models_dir, model_id)
    os.makedirs(save_dir)

    try:
        return MODEL_SUBCLASSES[model_name].create(model_id, save_dir, variables, model_config_dict, device=device)
    except KeyError as e:
        raise ModelClassNotFound() from e


def load_model(model_id: str, models_dir: str, device: Union[str, int]) -> IModel:
    """
    Loads an instance of an implementation of the `Model` class.

    Args:
        model_id (str): String corresponding to model's id.
        models_dir (str): Directory where mnodel information is saved.

    Returns:
        Deseralized instance of concrete implementation of `Model` class.
    """
    model_type_filepath = os.path.join(models_dir, "model_type.txt")
    with open(model_type_filepath, encoding="utf-8") as f:
        model_name = f.read()

    try:
        return MODEL_SUBCLASSES[model_name].load(model_id, models_dir, device)
    except KeyError as e:
        raise ModelClassNotFound() from e


def create_set_encoder(set_encoder_type: str, kwargs: dict) -> SetEncoderBaseModel:
    """
    Create a set encoder instance.

    Args:
        set_encoder_type (str): type of set encoder to create
        kwargs (dict): keyword arguments to pass to the set encoder constructor

    """
    # Create a set encoder instance.
    set_encoder_type_map: Dict[str, Type[SetEncoderBaseModel]] = {
        "default": PointNet,
        "sparse": SparsePointNet,
        "transformer": TransformerSetEncoder,
    }
    return set_encoder_type_map[set_encoder_type](**kwargs)
