import torch
from tensordict import TensorDict

from causica.functional_relationships.functional_relationships import FunctionalRelationships
from causica.nn import DECIEmbedNN


class DECIEmbedFunctionalRelationships(FunctionalRelationships):
    """
    This is a `FunctionalRelationsips` that wraps the `DECIEmbedNN` module.
    """

    def __init__(
        self,
        shapes: dict[str, torch.Size],
        embedding_size: int,
        out_dim_g: int,
        num_layers_g: int,
        num_layers_zeta: int,
    ) -> None:
        super().__init__(shapes=shapes)

        self.nn = DECIEmbedNN(self.stacked_key_masks, embedding_size, out_dim_g, num_layers_g, num_layers_zeta)

    def forward(self, samples: TensorDict, graphs: torch.Tensor) -> TensorDict:
        return self.tensor_to_td(self.nn(self.tensor_to_td.inv(samples), graphs))
