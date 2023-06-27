import abc
from typing import Any

import torch
from tensordict import TensorDict

from causica.distributions.transforms import TensorToTensorDictTransform


class FunctionalRelationships(abc.ABC, torch.nn.Module):
    def __init__(self, shapes: dict[str, torch.Size]) -> None:
        """_summary_

        Args:
            shapes: Dict of node shapes (how many dimensions a node has)
                Order corresponds to the order in graph(s).
        """
        super().__init__()
        self.shapes = shapes
        # create a transform for mapping tensors to tensordicts
        self.tensor_to_td = TensorToTensorDictTransform(shapes)
        # this needs to be registered to the module, and register buffer doesn't work
        self.stacked_key_masks = torch.nn.Parameter(self.tensor_to_td.stacked_key_masks(), requires_grad=False)

    def set_extra_state(self, state: dict[str, Any]):
        self.shapes = state.pop("shapes")
        self.tensor_to_td = TensorToTensorDictTransform(self.shapes)

    def get_extra_state(self) -> dict[str, Any]:
        return {"shapes": self.shapes}

    @abc.abstractmethod
    def forward(self, samples: TensorDict, graphs: torch.Tensor) -> TensorDict:
        """Calculates the predictions of the children from parents.

        Args:
            samples: dictionary of variable samples of shape sample_shape + [node shape]
            graphs: tensor of shape batch_shape + [nodes, nodes]

        Returns:
            Dictionary of torch.Tensors of shape sample_shape + batch_shape + [node shape]
        """
