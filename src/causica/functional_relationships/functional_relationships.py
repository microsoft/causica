import abc

import torch
from tensordict import TensorDict

from causica.distributions.transforms import TensorToTensorDictTransform


class FunctionalRelationships(abc.ABC, torch.nn.Module):
    def __init__(self, shapes: dict[str, torch.Size], batch_shape: torch.Size = torch.Size([])) -> None:
        """Base class for functional relationships.

        Args:
            shapes: Dict of node shapes (how many dimensions a node has)
                Order corresponds to the order in graph(s).
        """
        super().__init__()
        self.batch_shape = batch_shape
        self.shapes = shapes
        # create a transform for mapping tensors to tensordicts
        self.tensor_to_td = TensorToTensorDictTransform(shapes)
        # this needs to be registered to the module, and register buffer doesn't work
        self.stacked_key_masks = torch.nn.Parameter(self.tensor_to_td.stacked_key_masks(), requires_grad=False)

    @abc.abstractmethod
    def forward(self, samples: TensorDict, graphs: torch.Tensor) -> TensorDict:
        """Calculates the predictions of the children from parents.

        Functional relationships expect samples to have a batch shape in order: samples_shape, functions_shape,
        graphs_shape. The graphs are expected have a matching batch shape graphs_shape. This then applies the functional
        relationship to each sample using the corresponding function and graph. This allows for batched processing of
        different functions (e.g. interventions) or graphs.

        Args:
            samples: dictionary of variable samples of shape batch_size_x + batch_size_f + batch_shape_g + [node shape].
            graphs: tensor of shape batch_size_g + [nodes, nodes]

        Returns:
            Dictionary of torch.Tensors of shape batch_size_x + batch_size_f + batch_shape_g + [node shape]
        """
