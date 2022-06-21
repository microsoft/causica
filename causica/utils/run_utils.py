import os
from glob import glob
from time import gmtime, strftime
from typing import List, Optional, Tuple


def find_local_model_dir(model_dir: str, model_file_name: str = "model.pt") -> Tuple[str, str]:
    """
    Find the specific model directory within model_dir.

    Returns:
        model_dir: Full path to directory containing model.pt
        model_id: Unique ID for the model found.
    """
    all_model_dirs = find_all_model_dirs([model_dir], model_file_name=model_file_name)
    if not all_model_dirs:
        raise FileNotFoundError
    if len(all_model_dirs) > 1:
        raise ValueError("There are multiple model directories here, so the request is ambiguous")
    return all_model_dirs[0], os.path.basename(all_model_dirs[0])


def find_all_model_dirs(parent_dirs: List[str], model_file_name: str = "model.pt") -> List[str]:
    """
    List subdirectories of parent_dirs containing models.
    TODO this will include marginal/dep networks, which probably we do not want
    """

    model_dirs = []
    assert len(parent_dirs) == len(set(parent_dirs))
    for d in parent_dirs:
        if not os.path.exists(d):
            raise FileNotFoundError(d)
        model_files = glob(f"{d}/**/{model_file_name}", recursive=True)
        model_dirs.extend([os.path.dirname(f) for f in model_files])

    # If a parent/ancestor directory is also a model_dir, then d is probably marginal or dependency net of a VAEM,
    # and we don't want to return it.
    # TODO: this is not nice
    model_dirs = [d for d in model_dirs if not any(d.startswith(d2 + os.path.sep) for d2 in model_dirs)]

    return model_dirs


def create_models_dir(output_dir: str, name: Optional[str] = None) -> str:
    dir_name = strftime("%Y-%m-%d_%H%M%S", gmtime())
    if name:
        dir_name = f"{name}_{dir_name}"
    models_dir = os.path.join(output_dir, dir_name, "models")
    os.makedirs(models_dir)
    return models_dir
