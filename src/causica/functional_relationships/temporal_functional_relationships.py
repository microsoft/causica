import torch
from tensordict import TensorDict

from causica.functional_relationships.functional_relationships import FunctionalRelationships
from causica.nn import TemporalEmbedNN


class TemporalEmbedFunctionalRelationships(FunctionalRelationships):
    """This is a `FunctionalRelationsips` that wraps the `RhinoEmbedNN` module."""

    def __init__(
        self,
        shapes: dict[str, torch.Size],
        embedding_size: int,
        out_dim_g: int,
        num_layers_g: int,
        num_layers_zeta: int,
        context_length: int,
    ) -> None:
        """
        Args:
            shapes: Dictionary of shapes of the input tensors.
            embedding_size: See `TemporalEmbedNN`.
            out_dim_g: See `TemporalEmbedNN`.
            num_layers_g: See `TemporalEmbedNN`.
            num_layers_zeta: See `TemporalEmbedNN`.
            context_length: See `TemporalEmbedNN`.
        """
        super().__init__(shapes=shapes)
        self.nn = TemporalEmbedNN(
            group_mask=self.stacked_key_masks,
            embedding_size=embedding_size,
            out_dim_l=out_dim_g,
            num_layers_l=num_layers_g,
            num_layers_zeta=num_layers_zeta,
            context_length=context_length,
        )

    def forward(self, samples: TensorDict, graphs: torch.Tensor) -> TensorDict:
        return self.tensor_to_td(self.nn(self.tensor_to_td.inv(samples), graphs))
