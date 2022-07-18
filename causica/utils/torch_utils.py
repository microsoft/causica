import random
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.distributions as dist
from scipy.sparse import csr_matrix, issparse
from torch.nn import Dropout, LayerNorm, Linear, Module, Sequential
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler, TensorDataset

from ..utils.helper_functions import to_tensors


def set_random_seeds(seed):
    """
    Set random seeds for Torch, Numpy and Python, as well as Torch reproducibility settings.
    """
    if isinstance(seed, list) and len(seed) == 1:
        seed = seed[0]

    # PyTorch settings to ensure reproducibility - see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_torch_device(device_id: Union[int, str, torch.device] = 0) -> torch.device:
    """
    Get a torch device:
        - If CUDA is available will return a CUDA device (optionally can specify it to be the
          `device_id`th CUDA device), otherwise "cpu".
        - If 'gpu' is specified, default to the first GPU ('cuda:0')
        - Can request a CPU by providing a device id of 'cpu' or -1.

    Args:
        device_id (int, str, or torch.device): The ID of a CUDA device if more than one available on the system.
        Defaults to 0, which means GPU if it's available.

    Returns:
        :obj:`torch.device`: The available device.
    """
    # If input is already a Torch device, then return it as-is.
    if isinstance(device_id, torch.device):
        return device_id
    elif device_id in (-1, "cpu"):
        return torch.device("cpu")
    elif torch.torch.cuda.is_available():
        if device_id == "gpu":
            device_id = 0
        return torch.device(f"cuda:{device_id}")
    else:
        return torch.device("cpu")


class resBlock(Module):
    """
    Wraps an nn.Module, adding a skip connection to it.
    """

    def __init__(self, block: Module):
        """
        Args:
            block: module to which skip connection will be added. The input dimension must match the output dimension.
        """
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


def generate_fully_connected(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    non_linearity: Optional[Type[Module]],
    activation: Optional[Type[Module]],
    device: torch.device,
    p_dropout: float = 0.0,
    init_method: str = "default",
    normalization: Optional[Type[LayerNorm]] = None,
    res_connection: bool = False,
) -> Module:
    """
    Generate a fully connected network.

    Args:
        input_dim: Int. Size of input to network.
        output_dim: Int. Size of output of network.
        hidden_dims: List of int. Sizes of internal hidden layers. i.e. [a, b] is three linear layers with shapes (input_dim, a), (a, b), (b, output_dim)
        non_linearity: Non linear activation function used between Linear layers.
        activation: Final layer activation to use.
        device: torch device to load weights to.
        p_dropout: Float. Dropout probability at the hidden layers.
        init_method: initialization method
        normalization: Normalisation layer to use (batchnorm, layer norm, etc). Will be placed before linear layers, excluding the input layer.
        res_connection : Whether to use residual connections where possible (if previous layer width matches next layer width)

    Returns:
        Sequential object containing the desired network.
    """
    layers: List[Module] = []

    prev_dim = input_dim
    for idx, hidden_dim in enumerate(hidden_dims):

        block: List[Module] = []

        if normalization is not None and idx > 0:
            block.append(normalization(prev_dim).to(device))
        block.append(Linear(prev_dim, hidden_dim).to(device))

        if non_linearity is not None:
            block.append(non_linearity())
        if p_dropout != 0:
            block.append(Dropout(p_dropout))

        if res_connection and (prev_dim == hidden_dim):
            layers.append(resBlock(Sequential(*block)))
        else:
            layers.append(Sequential(*block))
        prev_dim = hidden_dim

    if normalization is not None:
        layers.append(normalization(prev_dim).to(device))
    layers.append(Linear(prev_dim, output_dim).to(device))

    if activation is not None:
        layers.append(activation())

    fcnn = Sequential(*layers)
    if init_method != "default":
        fcnn.apply((lambda x: alternative_initialization(x, init_method=init_method)))
    return fcnn


def alternative_initialization(module: Module, init_method: str) -> None:
    if isinstance(module, torch.nn.Linear):
        if init_method == "xavier_uniform":
            torch.nn.init.xavier_uniform_(module.weight)
        elif init_method == "xavier_normal":
            torch.nn.init.xavier_normal_(module.weight)
        elif init_method == "uniform":
            torch.nn.init.uniform_(module.weight)
        elif init_method == "normal":
            torch.nn.init.normal_(module.weight)
        else:
            return
        torch.nn.init.zeros_(module.bias)


class CrossEntropyLossWithConvert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self, prediction, actual):
        return self._loss(prediction, actual.long())


def create_dataloader(
    *arrays: Union[np.ndarray, csr_matrix],
    batch_size: int,
    iterations: int = -1,
    sample_randomly: bool = True,
    dtype: torch.dtype = torch.float,
    device: torch.device = torch.device("cpu"),
) -> DataLoader:
    """
    Device specifies the device on which the TensorDataset is created. This should be CPU in most cases, as we
    typically do not wish to store the whole dataset on the GPU.
    """
    assert len(arrays) > 0
    dataset: Dataset
    if issparse(arrays[0]):
        assert all(issparse(arr) for arr in arrays)
        # TODO: To fix type error need to cast arrays from Tuple[Union[ndarray, csr_matrix]] to Tuple[csr_matrix],
        # but MyPy doesn't seem to detect it when I do this.
        dataset = SparseTensorDataset(*arrays, dtype=dtype, device=device)  # type: ignore
    else:
        assert all(not issparse(arr) for arr in arrays)
        dataset = TensorDataset(*to_tensors(*arrays, dtype=dtype, device=device))

    row_count = arrays[0].shape[0]
    max_iterations = np.ceil(row_count / batch_size)
    if iterations > max_iterations:
        iterations = -1

    if sample_randomly:
        if iterations == -1:
            # mypy throws an error when using a pytorch Dataset for the pytorch RandomSampler. This seems to be an issue in pytorch typing.
            sampler: Sampler = RandomSampler(dataset)  # type: ignore
        else:
            sampler = RandomSampler(dataset, replacement=True, num_samples=iterations * batch_size)  # type: ignore
    else:
        sampler = SequentialSampler(dataset)  # type: ignore

    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=True)
    return dataloader


class SparseTensorDataset(Dataset):
    """
    Custom dataset class which takes in a sparse matrix (assumed to be efficiently indexable row-wise, ie csr) and
    returns dense tensors containing requested rows. Ensures that the large matrices are kept sparse at all times,
    and only converted to dense matrices one minibatch at a time.
    """

    def __init__(
        self,
        *matrices: Tuple[csr_matrix, ...],
        dtype: torch.dtype = torch.float,
        device: torch.device,
    ):
        self._matrices = matrices
        self._dtype = dtype
        self.device = device

    def __getitem__(self, idx):
        data_rows = tuple(
            torch.as_tensor(
                matrix[idx, :].toarray().squeeze(axis=0),
                dtype=self._dtype,
                device=self.device,
            )
            for matrix in self._matrices
        )
        return data_rows

    def __len__(self):
        return self._matrices[0].shape[0]


class LinearModel:
    def __init__(self):
        """
        Simple linear regression model learnt using a Gaussian prior
        """
        self.w = None
        self.posterior_prec = None

    def fit(self, features: torch.Tensor, targets: torch.Tensor, prior_precision: float = 1):
        """
        Learn weights from data using MAP inference
        Args:
            features: (Npoints x Nfeatures) tensor with training features
            targets: (Npoints,) tensor
            prior_precision: Precision of an isotropic Gaussian prior (also known as ridge regulariser)
        Returns:
            None
        """
        assert len(targets) == features.shape[0]
        assert len(features.shape) == 2

        self.posterior_prec = (
            features.T @ features
            + torch.eye(features.shape[1], dtype=features.dtype, device=features.device) * prior_precision
        )

        self.w = torch.linalg.solve(self.posterior_prec, features.T) @ targets

    def predict(self, features: torch.Tensor, compute_covariance=False):
        """
        Make predictions
        Args:
            features: (Npoints x Nfeatures) tensor containing test features
            compute_covariance: whether to compute and return a covariance matrix over test targets
        Returns:
            pred_mu: a (Npoints) tensor containing predicted values for the test points
            pred_cov: A (Npoints x Npoints) covariance matrix if compute_covariance is True, else None
        """
        assert len(self.w) == features.shape[1]
        assert self.w is not None, "model must be fit before it can make predictions"

        pred_mu = features @ self.w
        if compute_covariance:
            pred_cov = features @ torch.linalg.solve(self.posterior_prec, features.T)
        else:
            pred_cov = None

        return pred_mu, pred_cov


class MultiROFFeaturiser:
    def __init__(
        self, rff_n_features: int, lengthscale: Union[int, float, List[float], Tuple[float, ...]] = (1e-1, 0.5)
    ):
        """
        Random orthogonal fourier featuriser (https://proceedings.neurips.cc/paper/2016/file/53adaf494dc89ef7196d73636eb2451b-Paper.pdf) implementing sk-learn style fit and fit transform methods.
        Linear regression with RFF features approximates GP regression with an RBF kernel.

        Args:
            rff_n_features: size of the feature expansion
            lengthscale: of the equivalent RBF kernel if set to a float or int. If a 2 float tuple is specified,
                 the lengthscale of each random feature will be sampled randomly from a uniform between the
                 first and second tupples. This allows us to fit data for which we dont have prior lengthscale knowledge.
        """
        self.fitted = False
        self.rff_n_features = rff_n_features
        self.lengthscale = lengthscale
        if isinstance(self.lengthscale, (list, tuple)):
            assert len(self.lengthscale) == 2
            assert self.lengthscale[0] < self.lengthscale[1]
            assert self.lengthscale[0] > 0
        else:
            assert self.lengthscale > 0

    def fit(self, X: torch.Tensor):
        """
        Generate random coefficients from the shape of the training data
        Args:
            X: temsor of size (n_samples, n_features)
        """
        _, n_data_features = X.shape
        size = (n_data_features, n_data_features)
        # We compute random features in orthogonal blocks of size n_data_features
        n_stacks = int(np.ceil(self.rff_n_features / n_data_features))
        # As a result of orthogonality constraint we can have slightly more random features than specified
        rff_n_features = n_stacks * n_data_features

        random_weights_ = []
        for _ in range(n_stacks):
            # Iterate over stacks, building feature coefficients orthogonal in the dimension of the observations
            W = torch.randn(size, dtype=X.dtype, device=X.device)
            Q, _ = torch.qr(W, some=False)

            chi2 = (
                dist.Chi2(df=torch.tensor([n_data_features], dtype=X.dtype, device=X.device))
                .sample((n_data_features,))
                .sqrt()
                .squeeze()
            )
            if chi2.numel() > 1:
                chi2 = chi2.diag()
            else:
                chi2 = chi2.unsqueeze(0)
            SQ = Q @ chi2  # size (n_data_features, n_data_features)
            random_weights_ += [SQ]

        self.random_weights_ = torch.vstack(random_weights_).T  # Shape (n_data_features, rff_n_features)

        if isinstance(self.lengthscale, (list, tuple)):
            # Marginalise lengthscale over some uniform prior to provide a more flexible function class
            lengthscale = (
                torch.rand(1, n_stacks, dtype=X.dtype, device=X.device) * (self.lengthscale[1] - self.lengthscale[0])
                + self.lengthscale[0]
            )
            lengthscale = lengthscale.repeat_interleave(n_data_features).unsqueeze(0)
        else:
            # Scale feature coefficients by fixed lengthscale
            lengthscale = torch.tensor(self.lengthscale)

        self.random_weights_ /= lengthscale
        # Generate random bias term
        self.random_offset_ = torch.rand(rff_n_features, dtype=X.dtype, device=X.device) * np.pi * 2

        self.fitted = True

    def transform(self, X: torch.Tensor):
        """
        Apply random featurisation to observation X of of size (-, n_features)
        """
        assert self.fitted, "fit must be run before transform"
        output = X @ self.random_weights_
        output = torch.cos(output + self.random_offset_)
        output *= np.sqrt(2)
        return output / np.sqrt(self.rff_n_features)
