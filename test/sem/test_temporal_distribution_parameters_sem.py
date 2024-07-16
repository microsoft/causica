import torch
from tensordict import TensorDict

from causica.datasets.tensordict_utils import tensordict_shapes
from causica.sem.temporal_distribution_parameters_sem import split_lagged_and_instanteneous_values


def test_split_lagged_and_instanteneous_values():
    td = TensorDict(
        {
            "a": torch.rand(2, 3, 4),
            "b": torch.rand(2, 3, 5),
            "c": torch.rand(2, 3, 6),
            "d": torch.rand(2, 3, 7),
        },
        batch_size=2,
    )

    lagged, instantaneous = split_lagged_and_instanteneous_values(td)

    assert lagged.batch_size == instantaneous.batch_size == torch.Size([2])
    assert tensordict_shapes(lagged) == {
        "a": torch.Size([2, 4]),
        "b": torch.Size([2, 5]),
        "c": torch.Size([2, 6]),
        "d": torch.Size([2, 7]),
    }
    assert tensordict_shapes(instantaneous) == {
        "a": torch.Size([4]),
        "b": torch.Size([5]),
        "c": torch.Size([6]),
        "d": torch.Size([7]),
    }
