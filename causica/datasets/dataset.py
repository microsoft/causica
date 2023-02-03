import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from scipy.sparse import csr_matrix, issparse

from ..utils.io_utils import save_json
from .intervention_data import InterventionData
from .variables import Variables

T = Union[csr_matrix, np.ndarray]


class BaseDataset:
    def __init__(
        self,
        train_data: T,
        train_mask: T,
        val_data: Optional[T] = None,
        val_mask: Optional[T] = None,
        test_data: Optional[T] = None,
        test_mask: Optional[T] = None,
        variables: Optional[Variables] = None,
        data_split: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            train_data: Train data with shape (num_train_rows, input_dim).
            train_mask: Train mask indicating observed variables with shape (num_train_rows, input_dim). 1 is observed,
                        0 is un-observed.
            val_data: If given, validation data with shape (num_val_rows, input_dim).
            val_mask: If given, validation mask indicating observed variables with shape (num_val_rows, input_dim).
                      1 is observed, 0 is un-observed.
            test_data: If given, test data with shape (num_test_rows, input_dim).
            test_mask: If given, test mask indicating observed variables with shape (num_test_rows, input_dim). 1 is observed,
                       0 is un-observed.
            variables: If given, information about variables/features in the dataset.
            data_split: Dictionary record about how the row indices were split.
        """

        assert train_data.shape == train_mask.shape
        num_train_rows, num_cols = train_data.shape
        assert num_train_rows > 0
        if test_data is not None:
            assert test_mask is not None
            assert test_data.shape == test_mask.shape
            num_test_rows, num_test_cols = test_data.shape
            assert num_test_cols == num_cols
            assert num_test_rows > 0

        if val_data is not None:
            assert val_mask is not None
            assert val_data.shape == val_mask.shape
            assert val_data.shape[1] == num_cols
            assert val_data.shape[0] > 0
        self._train_data = train_data
        self._train_mask = train_mask
        self._val_data = val_data
        self._val_mask = val_mask
        self._test_data = test_data
        self._test_mask = test_mask
        self._variables = variables
        self._data_split = data_split

    def save_data_split(self, save_dir: str) -> None:
        """
        Save the data_split dictionary if it exists.
        Args:
            save_dir: Location to save the file.
        """
        if self._data_split is not None:
            save_json(self._data_split, os.path.join(save_dir, "data_split.json"))
        else:
            warnings.warn("Data split not available - it won't be saved.", UserWarning)

    @property
    def train_data_and_mask(self) -> Tuple[T, T]:
        return self._train_data, self._train_mask

    @property
    def val_data_and_mask(self) -> Tuple[Optional[T], Optional[T]]:
        return self._val_data, self._val_mask

    @property
    def test_data_and_mask(self) -> Tuple[Optional[T], Optional[T]]:
        return self._test_data, self._test_mask

    @property
    def variables(self) -> Optional[Variables]:
        return self._variables

    @property
    def data_split(self) -> Optional[Dict[str, Any]]:
        return self._data_split

    @classmethod
    def training_only(cls, data: T, mask: T):
        # Package up training data and mask in a valid dataset object
        return cls(data, mask, None, None, None, None, None)

    @property
    def has_val_data(self) -> bool:
        return self._val_data is not None and self._val_data.shape[0] > 0


class Dataset(BaseDataset):
    """
    Class to store dense train/val/test data and masks and variables metadata.
    Note that the data and masks provided by this class are read only.
    """

    def __init__(
        self,
        train_data: np.ndarray,
        train_mask: np.ndarray,
        val_data: Optional[np.ndarray] = None,
        val_mask: Optional[np.ndarray] = None,
        test_data: Optional[np.ndarray] = None,
        test_mask: Optional[np.ndarray] = None,
        variables: Optional[Variables] = None,
        data_split: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split)

        # Ensure that data and masks are immutable
        if not issparse(self._train_data):
            self._train_data.setflags(write=False)
            self._train_mask.setflags(write=False)
        if test_data is not None and not issparse(test_data):
            self._test_data = cast(np.ndarray, test_data)
            self._test_data.setflags(write=False)
            self._test_mask = cast(np.ndarray, test_mask)
            self._test_mask.setflags(write=False)

        if val_data is not None and not issparse(val_data):
            self._val_data = cast(np.ndarray, val_data)
            self._val_mask = cast(np.ndarray, val_mask)
            self._val_data.setflags(write=False)
            self._val_mask.setflags(write=False)

    def to_causal(
        self,
        adjacency_data: Optional[np.ndarray],
        subgraph_data: Optional[np.ndarray],
        intervention_data: Optional[List[InterventionData]],
        counterfactual_data: Optional[List[InterventionData]] = None,
    ):
        """
        Return the dag version of this dataset.
        """
        return CausalDataset(
            train_data=self._train_data,
            train_mask=self._train_mask,
            adjacency_data=adjacency_data,
            subgraph_data=subgraph_data,
            intervention_data=intervention_data,
            counterfactual_data=counterfactual_data,
            val_data=self._val_data,
            val_mask=self._val_mask,
            test_data=self._test_data,
            test_mask=self._test_mask,
            variables=self._variables,
            data_split=self._data_split,
        )

    def to_temporal(
        self,
        adjacency_data: Optional[np.ndarray],
        intervention_data: Optional[List[InterventionData]],
        transition_matrix: Optional[np.ndarray],
        counterfactual_data: Optional[List[InterventionData]],
        train_segmentation: Optional[List[Tuple[int, int]]] = None,
        test_segmentation: Optional[List[Tuple[int, int]]] = None,
        val_segmentation: Optional[List[Tuple[int, int]]] = None,
    ):
        """
        Return the dag version of this dataset.
        """
        return TemporalDataset(
            train_data=self._train_data,
            train_mask=self._train_mask,
            transition_matrix=transition_matrix,
            adjacency_data=adjacency_data,
            intervention_data=intervention_data,
            counterfactual_data=counterfactual_data,
            val_data=self._val_data,
            val_mask=self._val_mask,
            test_data=self._test_data,
            test_mask=self._test_mask,
            variables=self._variables,
            data_split=self._data_split,
            train_segmentation=train_segmentation,
            test_segmentation=test_segmentation,
            val_segmentation=val_segmentation,
        )

    def to_latent_confounded_causal(
        self,
        directed_adjacency_data: Optional[np.ndarray],
        directed_subgraph_data: Optional[np.ndarray],
        bidirected_adjacency_data: Optional[np.ndarray],
        bidirected_subgraph_data: Optional[np.ndarray],
        intervention_data: Optional[List[InterventionData]],
        counterfactual_data: Optional[List[InterventionData]] = None,
    ):
        """
        Return the admg version of this dataset.
        """
        return LatentConfoundedCausalDataset(
            train_data=self._train_data,
            train_mask=self._train_mask,
            directed_adjacency_data=directed_adjacency_data,
            directed_subgraph_data=directed_subgraph_data,
            bidirected_adjacency_data=bidirected_adjacency_data,
            bidirected_subgraph_data=bidirected_subgraph_data,
            intervention_data=intervention_data,
            counterfactual_data=counterfactual_data,
            val_data=self._val_data,
            val_mask=self._val_mask,
            test_data=self._test_data,
            test_mask=self._test_mask,
            variables=self._variables,
            data_split=self._data_split,
        )

    @property
    def train_data_and_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        # Add to avoid inconsistent type mypy error
        return self._train_data, self._train_mask


class SparseDataset(BaseDataset):
    """
    Class to store sparse train/val/test data and masks and variables metadata.
    """

    def __init__(
        self,
        train_data: csr_matrix,
        train_mask: csr_matrix,
        val_data: Optional[csr_matrix] = None,
        val_mask: Optional[csr_matrix] = None,
        test_data: Optional[csr_matrix] = None,
        test_mask: Optional[csr_matrix] = None,
        variables: Optional[Variables] = None,
        data_split: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split)
        # Declare types to avoid mypy error
        self._val_data: Optional[csr_matrix]
        self._val_mask: Optional[csr_matrix]
        self._test_data: Optional[csr_matrix]
        self._test_mask: Optional[csr_matrix]
        self._train_data: csr_matrix
        self._train_mask: csr_matrix

    def to_dense(self) -> Dataset:
        """
        Return the dense version of this dataset, i.e. all sparse data and mask arrays are transformed to dense.
        """
        val_data = self._val_data.toarray() if self._val_data is not None else None
        val_mask = self._val_mask.toarray() if self._val_mask is not None else None
        test_data = self._test_data.toarray() if self._test_data is not None else None
        test_mask = self._test_mask.toarray() if self._test_mask is not None else None
        return Dataset(
            self._train_data.toarray(),
            self._train_mask.toarray(),
            val_data,
            val_mask,
            test_data,
            test_mask,
            self._variables,
            self._data_split,
        )


class CausalDataset(Dataset):
    """
    Class to store the np.ndarray adjacency matrix and samples
     from the intervention distributions as attributes of the Dataset object.
    """

    def __init__(
        self,
        train_data: np.ndarray,
        train_mask: np.ndarray,
        adjacency_data: Optional[np.ndarray],
        subgraph_data: Optional[np.ndarray],
        intervention_data: Optional[List[InterventionData]],
        counterfactual_data: Optional[List[InterventionData]],
        val_data: Optional[np.ndarray] = None,
        val_mask: Optional[np.ndarray] = None,
        test_data: Optional[np.ndarray] = None,
        test_mask: Optional[np.ndarray] = None,
        variables: Optional[Variables] = None,
        data_split: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables, data_split)

        self._counterfactual_data = counterfactual_data
        self._intervention_data = intervention_data
        self._adjacency_data = adjacency_data
        self._subgraph_data = subgraph_data

    def get_adjacency_data_matrix(self) -> np.ndarray:
        """
        Return the np.ndarray dag adjacency matrix.
        """
        if self._adjacency_data is None:
            raise TypeError("Adjacency matrix is None. No adjacency matrix has been loaded.")
        return self._adjacency_data

    def set_adjacency_data_matrix(self, A: np.ndarray) -> None:
        """
        Externally set the np.ndarray dag adjacency matrix. If already set with a matrix, it will overwrite it
        """
        self._adjacency_data = A.copy()

    @property
    def has_adjacency_data_matrix(self) -> bool:
        """
        Returns: If the adjacency matrix is loaded
        """
        return self._adjacency_data is not None

    def get_known_subgraph_mask_matrix(self) -> np.ndarray:
        """
        Return the np.ndarray dag mask matrix.
        """
        if self._subgraph_data is None:
            raise TypeError("Adjacency matrix is None. No adjacency matrix has been loaded.")
        return self._subgraph_data

    def get_intervention_data(self) -> List[InterventionData]:
        """
        Return the list of interventions and samples from intervened distributions
        """
        if self._intervention_data is None:
            raise TypeError("Intervention data is None. No intervention data has been loaded.")
        return self._intervention_data

    def get_counterfactual_data(self) -> List[InterventionData]:
        """
        Return the list of interventions and samples for the counterfactual data
        """
        if self._counterfactual_data is None:
            raise TypeError("Counterfactual data is None. No counterfactual data has been loaded.")
        return self._counterfactual_data

    @property
    def has_counterfactual_data(self) -> bool:
        """
        Returns True if object has counterfactual data.
        """
        return self._counterfactual_data is not None


class TemporalDataset(CausalDataset):
    def __init__(
        self,
        train_data: np.ndarray,
        train_mask: np.ndarray,
        transition_matrix: Optional[np.ndarray],
        adjacency_data: Optional[np.ndarray],
        intervention_data: Optional[List[InterventionData]],
        counterfactual_data: Optional[List[InterventionData]],
        val_data: Optional[np.ndarray] = None,
        val_mask: Optional[np.ndarray] = None,
        test_data: Optional[np.ndarray] = None,
        test_mask: Optional[np.ndarray] = None,
        variables: Optional[Variables] = None,
        data_split: Optional[Dict[str, Any]] = None,
        train_segmentation: Optional[List[Tuple[int, int]]] = None,
        test_segmentation: Optional[List[Tuple[int, int]]] = None,
        val_segmentation: Optional[List[Tuple[int, int]]] = None,
    ) -> None:

        super().__init__(
            train_data=train_data,
            train_mask=train_mask,
            adjacency_data=None,
            subgraph_data=None,
            intervention_data=None,
            counterfactual_data=None,
            val_data=val_data,
            val_mask=val_mask,
            test_data=test_data,
            test_mask=test_mask,
            variables=variables,
            data_split=data_split,
        )

        self._counterfactual_data = counterfactual_data
        self._intervention_data = intervention_data
        self._adjacency_data = adjacency_data
        self._transition_matrix = transition_matrix
        # _train_segmentation stored as a list of tuples (start, end). For example, if the segmentation is
        # [(0, 10), (11, 15)], it means the first time series index starts with 0 to 10th row; and the second time series
        # index  with 11 to 15th row.
        self.train_segmentation = train_segmentation
        self._test_segmentation = test_segmentation
        self._val_segmentation = val_segmentation

    def get_transition_matrix(self) -> np.ndarray:
        """
        Return the np.ndarray transition matrix.
        """
        if self._transition_matrix is None:
            raise TypeError("Transition matrix is None. No transition matrix has been loaded.")
        return self._transition_matrix

    def get_val_segmentation(self) -> List[Tuple[int, int]]:
        """
        Return the list of tuples (start, end) for the validation data.
        """
        if self._val_segmentation is None:
            raise TypeError("Validation segmentation is None. No validation segmentation has been loaded.")
        return self._val_segmentation


class LatentConfoundedCausalDataset(CausalDataset):
    """Child class of CausalDataset that represents a causal dataset with latent confounding. This consists of both a
    directed adjacency matrix indicating direct causal interactions between observed variables, and a bidirected
    adjacency matrix indicating the presence of a latent confounder between pairs of observed variables."""

    def __init__(
        self,
        train_data: np.ndarray,
        train_mask: np.ndarray,
        directed_adjacency_data: Optional[np.ndarray],
        directed_subgraph_data: Optional[np.ndarray],
        bidirected_adjacency_data: Optional[np.ndarray],
        bidirected_subgraph_data: Optional[np.ndarray],
        intervention_data: Optional[List[InterventionData]],
        counterfactual_data: Optional[List[InterventionData]],
        val_data: Optional[np.ndarray] = None,
        val_mask: Optional[np.ndarray] = None,
        test_data: Optional[np.ndarray] = None,
        test_mask: Optional[np.ndarray] = None,
        variables: Optional[Variables] = None,
        data_split: Optional[Dict[str, Any]] = None,
    ) -> None:

        super().__init__(
            train_data=train_data,
            train_mask=train_mask,
            adjacency_data=None,
            subgraph_data=None,
            intervention_data=intervention_data,
            counterfactual_data=counterfactual_data,
            val_data=val_data,
            val_mask=val_mask,
            test_data=test_data,
            test_mask=test_mask,
            variables=variables,
            data_split=data_split,
        )

        self._directed_adjacency_data = directed_adjacency_data
        self._directed_subgraph_data = directed_subgraph_data
        self._bidirected_adjacency_data = bidirected_adjacency_data
        self._bidirected_subgraph_data = bidirected_subgraph_data

    @property
    def has_directed_adjacency_data_matrix(self) -> bool:
        """
        Returns: If the directed adjacency matrix is loaded
        """
        return self._directed_adjacency_data is not None

    def get_directed_adjacency_data_matrix(self) -> np.ndarray:
        """Returns the np.ndarray directed adjacency matrix."""
        if self._directed_adjacency_data is None:
            raise TypeError("Directed adjacency matrix is None. No directed adjacency matrix has been loaded.")
        return self._directed_adjacency_data

    def get_known_directed_subgraph_mask_matrix(self) -> np.ndarray:
        """Returns the np.ndarray mask matrix for the directed adjacency graph."""
        if self._directed_subgraph_data is None:
            raise TypeError("Directed mask matrix is None. No directed mask matrix has been loaded.")
        return self._directed_subgraph_data

    @property
    def has_bidirected_adjacency_data_matrix(self) -> bool:
        """
        Returns: If the bidirected adjacency matrix is loaded
        """
        return self._bidirected_adjacency_data is not None

    def get_bidirected_adjacency_data_matrix(self) -> np.ndarray:
        """Returns the np.ndarray directed adjacency matrix."""
        if self._bidirected_adjacency_data is None:
            raise TypeError("Bidirected adjacency matrix is None. No bidirected adjacency matrix has been loaded.")
        return self._bidirected_adjacency_data

    def get_known_bidirected_subgraph_mask_matrix(self) -> np.ndarray:
        """Returns the np.ndarray mask matrix for the bidirected adjacency graph."""
        if self._bidirected_subgraph_data is None:
            raise TypeError("Bidirected mask matrix is None. No bidirected mask matrix has been loaded.")
        return self._bidirected_subgraph_data
