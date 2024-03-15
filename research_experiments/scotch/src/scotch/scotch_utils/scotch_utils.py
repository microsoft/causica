import torch
from tensordict import TensorDict


def check_temporal_tensordict_shapes(num_timepoints: int, tds: TensorDict) -> bool:
    """Check that the shapes within the TensorDict are consistent with the number of time points."""
    return all(val.shape[len(tds.batch_size)] == num_timepoints for val in tds.values())


def temporal_tensordict_shapes(tds: TensorDict) -> dict[str, torch.Size]:
    """Return the shapes within the TensorDict without batch and time dimensions."""
    return {key: val.shape[len(tds.batch_size) + 1 :] for key, val in tds.items()}
