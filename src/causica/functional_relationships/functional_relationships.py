import abc
from typing import OrderedDict

import torch
from tensordict import TensorDict


class FunctionalRelationships(abc.ABC, torch.nn.Module):
    def __init__(self, variables: OrderedDict[str, torch.Size]) -> None:
        """_summary_

        Args:
            variables (OrderedDict[str, int]): Dict of node shapes (how many dimensions a variable has)
                Order corresponds to the order in graph(s).
        """
        super().__init__()
        self.num_nodes = len(variables)
        self.variables = variables
        self.output_shape = sum(variable.numel() for variable in variables.values())

        self.variable_masks = OrderedDict()
        last_idx = 0
        for name, shape in variables.items():
            mask = torch.zeros(self.output_shape, dtype=torch.bool)
            mask[last_idx : last_idx + shape.numel()] = True

            self.variable_masks[name] = mask
            last_idx += shape.numel()

    @abc.abstractmethod
    def forward(self, samples: TensorDict, graphs: torch.Tensor) -> TensorDict:
        """Calculates the predictions of the children from parents.

        Args:
            samples: dictionary of variable samples of shape sample_shape + [node shape]
            graphs: tensor of shape batch_shape + [nodes, nodes]

        Returns:
            Dictionary of torch.Tensors of shape sample_shape + batch_shape + [node shape]
        """
        pass


def sample_dict_to_tensor(sample_dict: TensorDict, variable_masks: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    """Converts a sample dictionary to a tensor."""
    return torch.cat([sample_dict[name] for name in variable_masks.keys()], dim=-1)


def tensor_to_sample_dict(sample_tensor: torch.Tensor, variable_masks: OrderedDict[str, torch.Tensor]) -> TensorDict:
    """Converts a tensor to a sample dictionary."""
    return TensorDict(
        {name: sample_tensor[..., mask] for name, mask in variable_masks.items()}, batch_size=sample_tensor.shape[:-1]
    )
