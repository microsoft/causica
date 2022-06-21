from abc import ABC, abstractmethod

import numpy as np


class ITextEmbeddingModel(ABC):
    """
    Interface for text embedding model:
    encode: Transforms text into embedding
    decode: Transforms embedding into text
    """

    @abstractmethod
    def encode(self, x: np.ndarray):
        """
        Transforms text into embedding

        Args:
            x: text data of shape (num_rows, num_features)
        """
        raise NotImplementedError()

    @abstractmethod
    def decode(self, y: np.ndarray):
        """
        Transforms embedding into text

        Args:
            y: embeddings data of shape (num_rows, total_num_latent_dimensions)
        """
        raise NotImplementedError()
