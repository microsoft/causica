import numpy as np
import torch

from causica.utils.helper_functions import (
    convert_dict_of_lists_to_ndarray,
    convert_dict_of_ndarray_to_lists,
    to_tensors,
)
from causica.utils.torch_utils import get_torch_device

cpu_torch_device = get_torch_device("cpu")


def test_convert_dict_of_lists_to_ndarray():
    dict_of_lists = {"a": [0, 1, 2], "b": [3, 4, 5], "c": [[6], [7], [8]]}

    dict_of_ndarrays = convert_dict_of_lists_to_ndarray(dict_of_lists)

    expected_dict = {"a": np.array([0, 1, 2]), "b": np.array([3, 4, 5]), "c": np.array([[6], [7], [8]])}

    np.testing.assert_equal(dict_of_ndarrays["a"], expected_dict["a"])
    np.testing.assert_equal(dict_of_ndarrays["b"], expected_dict["b"])
    np.testing.assert_equal(dict_of_ndarrays["c"], expected_dict["c"])

    dict_of_non_lists = {"a": "hello world", "b": 123, "c": 10.0, "d": {0, 1, 2}}

    processed_dict = convert_dict_of_lists_to_ndarray(dict_of_non_lists)

    assert processed_dict == dict_of_non_lists

    dict_mixed = {"a": True, "b": [[1, 2], [3, 4]]}

    processed_dict = convert_dict_of_lists_to_ndarray(dict_mixed)

    expected_dict = {"a": True, "b": np.array([[1, 2], [3, 4]])}

    assert expected_dict["a"] == processed_dict["a"]
    np.testing.assert_equal(expected_dict["a"], processed_dict["a"])


def test_convert_dict_of_ndarray_to_lists():
    dict_of_ndarrays = {"a": np.array([0, 1, 2]), "b": np.array([3, 4, 5]), "c": np.array([[6], [7], [8]])}

    dict_of_lists = {"a": [0, 1, 2], "b": [3, 4, 5], "c": [[6], [7], [8]]}

    processed_dict = convert_dict_of_ndarray_to_lists(dict_of_ndarrays)

    assert processed_dict == dict_of_lists


def test_to_tensors_one_input():
    a = np.ones((3, 4))

    # _to_tensors() returns a tuple so we need to explicitly unpack a single value with a comma
    (tensor_a,) = to_tensors(a, device=cpu_torch_device)
    assert isinstance(tensor_a, torch.Tensor)

    assert a.shape == tensor_a.shape
    assert tensor_a.dtype == torch.float


def test_to_tensors_two_inputs():
    a = np.ones((3, 4))
    b = np.ones((3, 4))

    tensor_a, tensor_b = to_tensors(a, b, device=cpu_torch_device)
    assert isinstance(tensor_a, torch.Tensor)
    assert isinstance(tensor_b, torch.Tensor)

    assert a.shape == tensor_a.shape
    assert b.shape == tensor_b.shape

    assert tensor_a.dtype == torch.float
    assert tensor_b.dtype == torch.float


def test_to_tensors_specify_dtype():
    a = np.ones((3, 4))
    b = np.ones((3, 4))

    tensor_a, tensor_b = to_tensors(a, b, device=cpu_torch_device, dtype=torch.int)
    assert isinstance(tensor_a, torch.Tensor)
    assert isinstance(tensor_b, torch.Tensor)

    assert a.shape == tensor_a.shape
    assert b.shape == tensor_b.shape

    assert tensor_a.dtype == torch.int
    assert tensor_b.dtype == torch.int
