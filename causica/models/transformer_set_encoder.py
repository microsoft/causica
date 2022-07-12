import math
from enum import Enum
from typing import Optional, cast

import torch

from .feature_embedder import FeatureEmbedder
from .set_encoder_base_model import SetEncoderBaseModel


class TransformerSetEncoder(SetEncoderBaseModel):
    """
    Embeds features using a FeatureEmbedder, then generates a set embedding using the SetTransformer.
    Note that empty sets are not sent to SetTransformer and are assigned a default empty set embedding instead.
    """

    feature_embedder_class = FeatureEmbedder

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        set_embedding_dim: int,
        metadata: Optional[torch.Tensor],
        device: torch.device,
        multiply_weights: bool = True,
        include_all_vars: bool = False,
        **kwargs
    ):
        """
        Args:
            input_dim: Dimension of input data to embedding model.
            embedding_dim: Dimension of embedding for each feature.
            set_embedding_dim: Dimension of output set embedding.
            metadata: Optional torch tensor. Each row represents a feature and each column
                is a metadata dimension for the feature. Shape (input_dim, metadata_dim).
            device: Torch device to use.
            multiply_weights: Boolean. Whether or not to take the product of x with embedding
                weights when feeding through. Defaults to True.
            include_all_vars: Whether all variables, observed and unobserved, should be included when creating the set embedding.
                If True, the data is multiplied by mask, to make sure that the unobserved variables are zeroed out.
        """
        super().__init__(input_dim, embedding_dim, set_embedding_dim, device)
        self._include_all_vars = include_all_vars
        if self._include_all_vars:
            assert not multiply_weights

        self.feature_embedder = self.feature_embedder_class(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            metadata=metadata,
            device=device,
            multiply_weights=multiply_weights,
        )
        self._input_embedding_dim = self.feature_embedder.output_dim
        self._set_embedding_dim = set_embedding_dim

        self._set_transformer = SetTransformer(
            input_embedding_dim=self._input_embedding_dim, set_embedding_dim=self._set_embedding_dim, **kwargs
        )

        self._empty_set_embedding = torch.zeros((self._set_embedding_dim,), device=self.device, requires_grad=False)
        self.to(device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).
            mask: Mask tensor with shape (batch_size, input_dim), 1 is observed, 0 is unobserved.
        Returns:
            set_embedding: Embedded output tensor with shape (batch_size, set_embedding_dim).
        """
        batch_size, _ = x.shape

        if self._include_all_vars:
            # Zero imputed masked data
            x = x * mask

            feature_embedded_x = self.feature_embedder(x)  # Shape (batch_size * input_dim, input_embedding_dim)
            feature_embedded_x = feature_embedded_x.reshape(
                (batch_size, self._input_dim, self._input_embedding_dim)
            )  # Shape (batch_size, input_dim, input_embedding_dim)

            # This channel held the embedding bias, which we don't need, so we can use it to hold the mask
            feature_embedded_x[:, :, -1] = mask

            set_embedding = self._set_transformer(
                x=feature_embedded_x, mask=None
            )  # Shape (batch_size, set_embedding_dim)

        else:
            # Only nonempty sets of features are passed to the set transformer
            is_empty_set = torch.all(mask == 0, dim=1)  # Booleans of shape (batch_size,)
            is_nonempty_set = torch.logical_not(is_empty_set)  # Booleans of shape (batch_size,)

            set_embedding = torch.empty((batch_size, self._set_embedding_dim), device=self.device)
            set_embedding[is_empty_set, :] = self._empty_set_embedding

            if not torch.all(is_empty_set):
                x = x[is_nonempty_set, :]
                mask = mask[is_nonempty_set, :]
                reduced_batch_size, _ = x.shape

                feature_embedded_x = self.feature_embedder(
                    x
                )  # Shape (reduced_batch_size * input_dim, input_embedding_dim)
                feature_embedded_x = feature_embedded_x.reshape(
                    (reduced_batch_size, self._input_dim, self._input_embedding_dim)
                )  # Shape (reduced_batch_size, input_dim, input_embedding_dim)
                nonempty_set_embedding = self._set_transformer(
                    feature_embedded_x, mask
                )  # Shape (reduced_batch_size, set_embedding_dim)
                set_embedding[is_nonempty_set, :] = nonempty_set_embedding

        return set_embedding


class SetTransformer(torch.nn.Module):
    """
    The Set Transformer model https://arxiv.org/abs/1810.00825

    Generates an embedding from a set of features using several blocks of self attention
    and pooling by attention.
    """

    class MultiheadInitType(Enum):
        xavier = 0
        kaiming = 1

    class ElementwiseTransformType(Enum):
        single = 0
        double = 1

    def __init__(
        self,
        input_embedding_dim: int,
        set_embedding_dim: int,
        transformer_embedding_dim: Optional[int] = None,
        num_heads: int = 1,
        num_blocks: int = 2,
        num_seed_vectors: int = 1,
        use_isab: bool = False,
        num_inducing_points: Optional[int] = None,
        multihead_init_type: str = "xavier",
        use_layer_norm: bool = True,
        elementwise_transform_type: str = "single",
        use_elementwise_transform_pma: bool = True,
        **_
    ):
        """
        Args:
            input_embedding_dim: Dimension of the input data, the embedded features.
            set_embedding_dim: Dimension of the output data, the set embedding.
            transformer_embedding_dim: Embedding dimension to be used in the set transformer blocks.
            num_heads: Number of heads in each multi-head attention block.
            num_blocks: Number of SABs in the model.
            num_seed_vectors: Number of seed vectors used in the pooling block (PMA).
            use_isab: Should ISAB blocks be used instead of SAB blocks.
            num_inducing_points: Number of inducing points.
            multihead_init_type: How linear layers in torch.nn.MultiheadAttention are initialised. Valid options are "xavier" and "kaiming".
            use_layer_norm: Whether layer normalisation should be used in MAB blocks.
            elementwise_transform_type: What version of the elementwise transform (rFF) should be used. Valid options are "single" and "double".
            use_elementwise_transform_pma: Whether an elementwise transform (rFF) should be used in the PMA block.
        """
        super().__init__()
        self._transform_input_dimension = transformer_embedding_dim is not None
        if self._transform_input_dimension:
            transformer_embedding_dim = cast(int, transformer_embedding_dim)
            self._input_dimension_transform = torch.nn.Linear(input_embedding_dim, transformer_embedding_dim)

        if transformer_embedding_dim is None:
            transformer_embedding_dim = input_embedding_dim

        if use_isab:
            if num_inducing_points is None:
                raise ValueError("Number of inducing points must be defined to use ISAB")
            self._sabs = torch.nn.ModuleList(
                [
                    ISAB(
                        embedding_dim=transformer_embedding_dim,
                        num_heads=num_heads,
                        num_inducing_points=num_inducing_points,
                        multihead_init_type=self.MultiheadInitType[multihead_init_type],
                        use_layer_norm=use_layer_norm,
                        elementwise_transform_type=self.ElementwiseTransformType[elementwise_transform_type],
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:
            self._sabs = torch.nn.ModuleList(
                [
                    SAB(
                        embedding_dim=transformer_embedding_dim,
                        num_heads=num_heads,
                        multihead_init_type=self.MultiheadInitType[multihead_init_type],
                        use_layer_norm=use_layer_norm,
                        elementwise_transform_type=self.ElementwiseTransformType[elementwise_transform_type],
                    )
                    for _ in range(num_blocks)
                ]
            )

        self._pma = PMA(
            embedding_dim=transformer_embedding_dim,
            num_heads=num_heads,
            num_seed_vectors=num_seed_vectors,
            multihead_init_type=self.MultiheadInitType[multihead_init_type],
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=self.ElementwiseTransformType[elementwise_transform_type],
            use_elementwise_transform_pma=use_elementwise_transform_pma,
        )
        self._output_dimension_transform = torch.nn.Linear(
            transformer_embedding_dim * num_seed_vectors, set_embedding_dim
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Embedded features tensor with shape (batch_size, set_size, input_embedding_dim).
            mask: Mask tensor with shape (batch_size, set_size), 1 is observed, 0 is unobserved.
        Returns:
            set_embedding: Set embedding tensor with shape (batch_size, set_embedding_dim).
        """
        self._validate_input(x, mask)
        batch_size, _, _ = x.shape

        if self._transform_input_dimension:
            x = self._input_dimension_transform(x)  # Shape (batch_size, set_size, transformer_embedding_dim)
        for sab in self._sabs:
            x = sab(x, mask)  # Shape (batch_size, set_size, transformer_embedding_dim)
        x = self._pma(x, mask)  # Shape (batch_size, num_seed_vectors, transformer_embedding_dim)
        x = x.reshape((batch_size, -1))  # Shape (batch_size, num_seed_vectors * transformer_embedding_dim)
        set_embedding = self._output_dimension_transform(x)  # Shape (batch_size, set_embedding_dim)

        return set_embedding

    def _validate_input(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> None:
        """
        Args:
            x: Embedded features tensor with shape (batch_size, set_size, input_embedding_dim).
            mask: Mask tensor with shape (batch_size, set_size), 1 is observed, 0 is unobserved.
        """
        assert x.dim() == 3, "x should have shape (batch_size, set_size, transformer_embedding_dim)"

        # The batch size has to be greater than zero to be passed to MultiheadAttention torch module
        batch_size, _, _ = x.shape
        assert batch_size > 0, "Batch size has to be greater than zero"

        if mask is not None:
            assert mask.dim() == 2, "mask should have shape (batch_size, set_size)"

            # We do not allow empty sets because softmax in attention introduces NaNs for empty sets
            is_empty_set = torch.all(mask == 0, dim=1)
            assert not torch.any(is_empty_set), "SetTransformer does not accept empty sets"

    @classmethod
    def initialise_multihead(
        cls, multihead: torch.nn.MultiheadAttention, multihead_init_type: MultiheadInitType
    ) -> None:
        if multihead_init_type == cls.MultiheadInitType.xavier:  # AIAYN and torch.nn.MultiheadAttention default
            torch.nn.init.xavier_uniform_(multihead.in_proj_weight)
            torch.nn.init.constant_(multihead.in_proj_bias, 0.0)

            torch.nn.init.kaiming_uniform_(multihead.out_proj.weight, a=math.sqrt(5))
            torch.nn.init.constant_(multihead.out_proj.bias, 0.0)

        else:
            assert multihead_init_type == cls.MultiheadInitType.kaiming  # ST Implementation (torch.nn.Linear) default
            torch.nn.init.kaiming_uniform_(multihead.in_proj_weight, a=math.sqrt(5))
            # pylint: disable=protected-access
            in_proj_fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(multihead.in_proj_weight)
            in_proj_bound = 1 / math.sqrt(in_proj_fan_in)
            torch.nn.init.uniform_(multihead.in_proj_bias, -in_proj_bound, in_proj_bound)

            torch.nn.init.kaiming_uniform_(multihead.out_proj.weight, a=math.sqrt(5))
            out_proj_fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(multihead.out_proj.weight)
            out_proj_bound = 1 / math.sqrt(out_proj_fan_in)
            torch.nn.init.uniform_(multihead.out_proj.bias, -out_proj_bound, out_proj_bound)

    @classmethod
    def create_elementwise_transform(
        cls, embedding_dim: int, elementwise_transform_type: ElementwiseTransformType
    ) -> torch.nn.Sequential:
        if elementwise_transform_type == cls.ElementwiseTransformType.single:  # ST Implementation default
            return torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, embedding_dim),
                torch.nn.ReLU(),
            )

        else:
            assert elementwise_transform_type == cls.ElementwiseTransformType.double  # AIAYN Implementation default
            return torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, embedding_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(embedding_dim, embedding_dim),
            )


class MAB(torch.nn.Module):
    """
    Multihead Attention Block of the Set Transformer model.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        multihead_init_type: SetTransformer.MultiheadInitType,
        use_layer_norm: bool,
        elementwise_transform_type: SetTransformer.ElementwiseTransformType,
    ):
        """
        Args:
            embedding_dim: Dimension of the input data.
            num_heads: Number of heads.
            multihead_init_type: How linear layers in torch.nn.MultiheadAttention are initialised.
            use_layer_norm: Whether layer normalisation should be used in MAB blocks.
            elementwise_transform_type: What version of the elementwise transform (rFF) should be used.
        """
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"

        self._multihead = torch.nn.MultiheadAttention(embedding_dim, num_heads)
        SetTransformer.initialise_multihead(self._multihead, multihead_init_type)

        self._use_layer_norm = use_layer_norm
        if self._use_layer_norm:
            self._layer_norm_1 = torch.nn.LayerNorm(embedding_dim)
            self._layer_norm_2 = torch.nn.LayerNorm(embedding_dim)

        self._elementwise_transform = SetTransformer.create_elementwise_transform(
            embedding_dim, elementwise_transform_type
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        The key_mask enforces that only the selected values are attended to in multihead attention,
        but the output is generated for all query elements.
        Args:
            query: Query tensor with shape (batch_size, query_set_size, embedding_dim)
            key: Input tensor with shape (batch_size, key_set_size, embedding_dim) to be used as key and value.
            key_mask: Mask tensor with shape (batch_size, key_set_size), 1 is observed, 0 is unobserved.
                If None, everything is observed.
        Returns:
            output: Attention output tensor with shape (batch_size, query_set_size, embedding_dim).
        """

        # MultiheadAttention requires inputs with shape (set_size, batch_size, embedding_dim)
        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)

        # PyTorch MultiheadAttention uses opposite convention to Causica for masks: 'True' means 'hidden'.
        if key_mask is None:
            # Everything is observed.
            inverted_key_mask = None
        else:
            inverted_key_mask = torch.logical_not(key_mask)

        x = (
            query
            + self._multihead(query=query, key=key, value=key, key_padding_mask=inverted_key_mask, need_weights=False)[
                0
            ]
        )

        if self._use_layer_norm:
            x = self._layer_norm_1(x)
        x = x + self._elementwise_transform(x)
        if self._use_layer_norm:
            x = self._layer_norm_2(x)
        output = x.permute(1, 0, 2)

        return output


class SAB(torch.nn.Module):
    """
    Self Attention Block of the Set Transformer model.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        multihead_init_type: SetTransformer.MultiheadInitType,
        use_layer_norm: bool,
        elementwise_transform_type: SetTransformer.ElementwiseTransformType,
    ):
        """
        Args:
            embedding_dim: Dimension of the input data.
            num_heads: Number of heads.
            multihead_init_type: How linear layers in torch.nn.MultiheadAttention are initialised.
            use_layer_norm: Whether layer normalisation should be used in MAB blocks.
            elementwise_transform_type: What version of the elementwise transform (rFF) should be used.
        """
        super().__init__()
        self._mab = MAB(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            multihead_init_type=multihead_init_type,
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=elementwise_transform_type,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        The mask enforces that only the selected values are attended to in multihead attention,
        but the output is generated for all elements of x.
        Args:
            x: Input tensor with shape (batch_size, set_size, embedding_dim) to be used as query, key and value.
            mask: Mask tensor with shape (batch_size, set_size), 1 is observed, 0 is unobserved.
                If None, everything is observed.
        Returns:
            output: Attention output tensor with shape (batch_size, set_size, embedding_dim).
        """
        output = self._mab(x, x, mask)
        return output


class PMA(torch.nn.Module):
    """
    Pooling by Multihead Attention block of the Set Transformer model.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_seed_vectors: int,
        multihead_init_type: SetTransformer.MultiheadInitType,
        use_layer_norm: bool,
        elementwise_transform_type: SetTransformer.ElementwiseTransformType,
        use_elementwise_transform_pma: bool,
    ):
        """
        Args:
            embedding_dim: Dimension of the input data.
            num_heads: Number of heads.
            num_seed_vectors: Number of seed vectors.
            multihead_init_type: How linear layers in torch.nn.MultiheadAttention are initialised.
            use_layer_norm: Whether layer normalisation should be used in MAB blocks.
            elementwise_transform_type: What version of the elementwise transform (rFF) should be used.
            use_elementwise_transform_pma: Whether an elementwise transform (rFF) should be used in the PMA block.
        """
        super().__init__()
        self._mab = MAB(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            multihead_init_type=multihead_init_type,
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=elementwise_transform_type,
        )

        self._seed_vectors = torch.nn.Parameter(torch.Tensor(1, num_seed_vectors, embedding_dim))
        torch.nn.init.xavier_uniform_(self._seed_vectors)

        self._use_elementwise_transform_pma = use_elementwise_transform_pma
        if self._use_elementwise_transform_pma:
            self._elementwise_transform = SetTransformer.create_elementwise_transform(
                embedding_dim, elementwise_transform_type
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Seed vectors attend to the given values.
        The mask enforces that only the selected values are attended to in multihead attention.
        Args:
            x: Input tensor with shape (batch_size, set_size, embedding_dim) to be used as key and value.
            mask: Mask tensor with shape (batch_size, set_size), 1 is observed, 0 is unobserved.
                If None, everything is observed.
        Returns:
            output: Attention output tensor with shape (batch_size, num_seed_vectors, embedding_dim).
        """
        if self._use_elementwise_transform_pma:
            x = self._elementwise_transform(x)

        batch_size, _, _ = x.shape
        seed_vectors_repeated = self._seed_vectors.expand(batch_size, -1, -1)
        output = self._mab(seed_vectors_repeated, x, mask)

        return output


class ISAB(torch.nn.Module):
    """
    Inducing-point self attention block. This reduces memory use and compute time from O(N^2) to O(NM)
    where N is the number of features and M is the number of inducing points.
    Reference: https://arxiv.org/pdf/1810.00825.pdf

    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        num_inducing_points: int,
        multihead_init_type: SetTransformer.MultiheadInitType,
        use_layer_norm: bool,
        elementwise_transform_type: SetTransformer.ElementwiseTransformType,
    ):
        """
        Args:
            embedding_dim: Dimension of the input data.
            num_heads: Number of heads.
            num_inducing_points: Number of inducing points.
            multihead_init_type: How linear layers in torch.nn.MultiheadAttention are initialised.
            use_layer_norm: Whether layer normalisation should be used in MAB blocks.
            elementwise_transform_type: What version of the elementwise transform (rFF) should be used.
        """
        super().__init__()
        assert num_inducing_points is not None
        self._mab1 = MAB(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            multihead_init_type=multihead_init_type,
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=elementwise_transform_type,
        )
        self._mab2 = MAB(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            multihead_init_type=multihead_init_type,
            use_layer_norm=use_layer_norm,
            elementwise_transform_type=elementwise_transform_type,
        )
        self._inducing_points = torch.nn.Parameter(torch.Tensor(1, num_inducing_points, embedding_dim))
        torch.nn.init.xavier_uniform_(self._inducing_points)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        """
        The mask enforces that only the selected values are attended to in multihead attention,
        but the output is generated for all elements of x.
        Args:
            x: Input tensor with shape (batch_size, set_size, embedding_dim) to be used as query, key and value.
            mask: Mask tensor with shape (batch_size, set_size), 1 is observed, 0 is unobserved.
                If None, everything is observed.
        Returns:
            output: Attention output tensor with shape (batch_size, set_size, embedding_dim).
        """
        batch_size, _, _ = x.shape
        inducing_points = self._inducing_points.expand(
            batch_size, -1, -1
        )  # Shape (batch_size, num_inducing_points, embedding_dim)
        y = self._mab1(
            query=inducing_points, key=x, key_mask=mask
        )  # Shape (batch_size, num_inducing_points, embedding_dim)
        return self._mab2(query=x, key=y)  # Shape (batch_size, set_size, embedding_dim)
