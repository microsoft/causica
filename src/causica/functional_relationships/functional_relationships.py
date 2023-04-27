import abc
from typing import Any

import torch
from tensordict import TensorDict


class FunctionalRelationships(abc.ABC, torch.nn.Module):
    def __init__(self, variables: dict[str, torch.Size]) -> None:
        """_summary_

        Args:
            variables: Dict of node shapes (how many dimensions a variable has)
                Order corresponds to the order in graph(s).
        """
        super().__init__()
        self.num_nodes = len(variables)
        self.variables = variables
        self.output_shape = sum(variable.numel() for variable in variables.values())

        self.variable_masks = {}
        last_idx = 0
        for name, shape in variables.items():
            mask = torch.zeros(self.output_shape, dtype=torch.bool)
            mask[last_idx : last_idx + shape.numel()] = True

            self.variable_masks[name] = mask
            last_idx += shape.numel()

    def set_extra_state(self, state: dict[str, Any]):
        self.num_nodes = state.pop("num_nodes")
        self.variables = state.pop("variables")
        self.output_shape = state.pop("output_shape")

    def get_extra_state(self) -> dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "variables": self.variables,
            "output_shape": self.output_shape,
        }

    @abc.abstractmethod
    def forward(self, samples: TensorDict, graphs: torch.Tensor) -> TensorDict:
        """Calculates the predictions of the children from parents.

        Args:
            samples: dictionary of variable samples of shape sample_shape + [node shape]
            graphs: tensor of shape batch_shape + [nodes, nodes]

        Returns:
            Dictionary of torch.Tensors of shape sample_shape + batch_shape + [node shape]
        """


def sample_dict_to_tensor(sample_dict: TensorDict, variable_masks: dict[str, torch.Tensor]) -> torch.Tensor:
    """Converts a sample dictionary to a tensor."""
    return torch.cat([sample_dict[name] for name in variable_masks.keys()], dim=-1)


def tensor_to_sample_dict(sample_tensor: torch.Tensor, variable_masks: dict[str, torch.Tensor]) -> TensorDict:
    """Converts a tensor to a sample dictionary."""
    return TensorDict(
        {name: sample_tensor[..., mask] for name, mask in variable_masks.items()}, batch_size=sample_tensor.shape[:-1]
    )
