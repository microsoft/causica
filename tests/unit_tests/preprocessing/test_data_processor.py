import numpy as np
import pytest
from scipy.sparse import csr_matrix

from causica.datasets.variables import Variable, Variables
from causica.preprocessing.data_processor import DataProcessor


@pytest.fixture(scope="function")
def data_processor():
    variables = Variables(
        [
            Variable("binary_input", True, "binary", 0, 1),
            Variable("categorical_input", True, "categorical", 0, 3),
            Variable("numeric_input", True, "continuous", 3, 13),
            Variable("smaller_categorical_input", True, "categorical", 2, 4),
            Variable("numeric_target", False, "continuous", 2, 300),
        ]
    )
    return DataProcessor(variables)


@pytest.fixture(scope="function")
def data_processor_not_squashed():
    variables = Variables(
        [
            Variable("binary_input", True, "binary", 0, 1),
            Variable("categorical_input", True, "categorical", 0, 3),
            Variable("numeric_input", True, "continuous", 3, 13),
            Variable("smaller_categorical_input", True, "categorical", 2, 4),
            Variable("numeric_target", False, "continuous", 2, 300),
        ]
    )
    return DataProcessor(variables, unit_scale_continuous=False)


@pytest.fixture(scope="function")
def data_processor_standardize():
    variables = Variables(
        [
            Variable("binary_input", True, "binary", 0, 1),
            Variable("categorical_input", True, "categorical", 0, 3),
            Variable("numeric_input", True, "continuous", 3, 13),
            Variable("smaller_categorical_input", True, "categorical", 2, 4),
            Variable("numeric_target", False, "continuous", 2, 300),
        ]
    )
    return DataProcessor(variables, unit_scale_continuous=False, standardize_data_mean=True, standardize_data_std=True)


good_data = np.array([[0, 2, 3, 2, 40], [1, 1, 4, 4, 16]], dtype=float)
too_high_data = np.array([[0, 2, 3, 2, 40], [1, 1, 15, 4, 16]], dtype=float)
too_low_data = np.array([[0, 2, 3, 2, 40], [1, 1, 2, 4, 16]], dtype=float)
non_integer_binary_data = np.array([[0.1, 2, 3, 2, 40], [0.9, 1, 4, 4, 16]], dtype=float)
non_integer_categorical_data = np.array([[0, 1.1, 3, 2, 40], [1, 1, 4, 4, 16]], dtype=float)

processed_good_data = np.array(
    [
        [0, 0, 0, 1, 0, 0, 1, 0, 0, (40 - 2) / (300 - 2)],
        [1, 0, 1, 0, 0, (4 - 3) / (13 - 3), 0, 0, 1, (16 - 2) / (300 - 2)],
    ]
)

processed_good_data_not_squashed = np.array(
    [
        [0, 0, 0, 1, 0, 3, 1, 0, 0, 40],
        [1, 0, 1, 0, 0, 4, 0, 0, 1, 16],
    ]
)

good_mask = np.array([[0, 1, 1, 1, 1], [1, 0, 1, 0, 1]])
bad_mask_out_of_range = np.array([[2, 1, 1, 1, 1], [1, 1, 0, 0, 1]])
bad_mask_float = np.array([[1.1, 1, 1, 1, 0.1], [1, 1, 0, 0, 1]])

processed_good_mask = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 1]])


# pylint: disable=redefined-outer-name


@pytest.mark.parametrize(
    "data, mask, warning",
    [
        (
            np.array([[0, 0, 3, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            None,
        ),  # Valid data
        (
            np.array([[0, 0, 13.1, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            UserWarning,
        ),  # Too high cts, observed
        (
            np.array([[0, 0, 2.9, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            UserWarning,
        ),  # Too low cts, observed
        (
            np.array([[0, 0, 13.1, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]),
            None,
        ),  # Too high cts, unobserved
        (
            np.array([[0, 0, 2.9, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]),
            None,
        ),  # Too low cts, unobserved
    ],
)
def test_check_data_cts_errors(data_processor, data, mask, warning):
    if warning is not None:
        with pytest.warns(warning):
            data_processor.check_data(data, mask)
    else:
        data_processor.check_data(data, mask)


@pytest.mark.parametrize(
    "data, mask, error",
    [
        (
            np.array([[2, 0, 3, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            ValueError,
        ),  # Too high binary, observed
        (
            np.array([[0, 4, 3, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            ValueError,
        ),  # Too high categorical, observed
        (
            np.array([[-1, 0, 3, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            ValueError,
        ),  # Too low binary, observed
        (
            np.array([[0, -1, 3, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            ValueError,
        ),  # Too low categorical, observed
        (
            np.array([[0.5, 0, 3, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            AssertionError,
        ),  # Non-integer binary, observed
        (
            np.array([[0, 0.5, 3, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            AssertionError,
        ),  # Non-integer categorical, observed
        (
            np.array([[2, 0, 3, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            None,
        ),  # Too high binary, unobserved
        (
            np.array([[0, 4, 3, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[1, 0, 1, 1, 1], [1, 1, 1, 1, 1]]),
            None,
        ),  # Too high categorical, unobserved
        (
            np.array([[-1, 0, 3, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            None,
        ),  # Too low binary, unobserved
        (
            np.array([[0, -1, 3, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[1, 0, 1, 1, 1], [1, 1, 1, 1, 1]]),
            None,
        ),  # Too low categorical, unobserved
        (
            np.array([[0.5, 0, 3, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
            None,
        ),  # Non-integer binary, unobserved
        (
            np.array([[0, 0.5, 3, 3, 2], [0, 0, 3, 3, 2]]),
            np.array([[1, 0, 1, 1, 1], [1, 1, 1, 1, 1]]),
            None,
        ),  # Non-integer categorical, unobserved
    ],
)
def test_check_data_discrete_errors(data_processor, data, mask, error):
    if error is not None:
        with pytest.raises(error):
            data_processor.check_data(data, mask)
    else:
        data_processor.check_data(data, mask)


@pytest.mark.parametrize(
    "data, mask, warning",
    [
        (np.array([[3, 2], [3, 2]]), np.array([[1, 1], [1, 1]]), None),  # Valid data
        (
            np.array([[13.1, 2], [3, 2]]),
            np.array([[1, 1], [1, 1]]),
            UserWarning,
        ),  # Too high cts, observed
        (
            np.array([[2.9, 2], [3, 2]]),
            np.array([[1, 1], [1, 1]]),
            UserWarning,
        ),  # Too low cts, observed
        (
            np.array([[13.1, 2], [3, 2]]),
            np.array([[0, 1], [1, 1]]),
            None,
        ),  # Too high cts, unobserved
        (
            np.array([[2.9, 2], [3, 2]]),
            np.array([[0, 1], [1, 1]]),
            None,
        ),  # Too low cts, unobserved
    ],
)
def test_check_continuous_data(data_processor, data, mask, warning):
    lower = np.array([3, 2])
    upper = np.array([13, 300])
    epsilon = 1e-5
    if warning is not None:
        with pytest.warns(warning):
            data_processor.check_continuous_data(data, mask, lower, upper, epsilon)
    else:
        data_processor.check_continuous_data(data, mask, lower, upper, epsilon)


@pytest.mark.parametrize(
    "data, mask, error",
    [
        (np.array([[0, 0], [0, 0]]), np.array([[1, 1], [1, 1]]), None),  # Valid data
        (
            np.array([[2, 0], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            ValueError,
        ),  # Too high binary, observed
        (
            np.array([[0, 4], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            ValueError,
        ),  # Too high categorical, observed
        (
            np.array([[-1, 0], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            ValueError,
        ),  # Too low binary, observed
        (
            np.array([[0, -1], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            ValueError,
        ),  # Too low categorical, observed
        (
            np.array([[0.5, 0], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            AssertionError,
        ),  # Non-integer binary, observed
        (
            np.array([[0, 0.5], [0, 0]]),
            np.array([[1, 1], [1, 1]]),
            AssertionError,
        ),  # Non-integer categorical, observed
        (
            np.array([[2, 0], [0, 0]]),
            np.array([[0, 1], [1, 1]]),
            None,
        ),  # Too high binary, unobserved
        (
            np.array([[0, 4], [0, 0]]),
            np.array([[1, 0], [1, 1]]),
            None,
        ),  # Too high categorical, unobserved
        (
            np.array([[-1, 0], [0, 0]]),
            np.array([[0, 1], [1, 1]]),
            None,
        ),  # Too low binary, unobserved
        (
            np.array([[0, -1], [0, 0]]),
            np.array([[1, 0], [1, 1]]),
            None,
        ),  # Too low categorical, unobserved
        (
            np.array([[0.5, 0], [0, 0]]),
            np.array([[0, 1], [1, 1]]),
            None,
        ),  # Non-integer binary, unobserved
        (
            np.array([[0, 0.5], [0, 0]]),
            np.array([[1, 0], [1, 1]]),
            None,
        ),  # Non-integer categorical, unobserved
    ],
)
def test_check_discrete_data(data_processor, data, mask, error):
    lower = np.array([0, 0])
    upper = np.array([1, 3])
    epsilon = 1e-5
    if error is not None:
        with pytest.raises(error):
            data_processor.check_discrete_data(data, mask, lower, upper, epsilon)
    else:
        data_processor.check_discrete_data(data, mask, lower, upper, epsilon)


@pytest.mark.parametrize(
    "mask, error",
    [
        (good_mask, None),
        (bad_mask_out_of_range, ValueError),
        (bad_mask_float, ValueError),
    ],
)
def test_mask(data_processor, mask, error):
    if error is not None:
        with pytest.raises(error):
            data_processor.check_mask(mask)
    else:
        data_processor.check_mask(mask)


def test_process_data(data_processor):
    processed_data = data_processor.process_data(good_data)
    assert np.all(np.isclose(processed_data, processed_good_data))


def test_process_data_not_squashed(data_processor_not_squashed):
    processed_data = data_processor_not_squashed.process_data(good_data)
    assert np.all(np.isclose(processed_data, processed_good_data_not_squashed))


def test_process_data_standardize(data_processor_standardize):
    continuous_mask = np.zeros_like(good_data)
    continuous_mask[:, [2, 4]] = 1
    processed_data, processed_mask, *_ = data_processor_standardize.process_data_and_masks(good_data, continuous_mask)
    assert np.isclose(np.mean(processed_data[processed_mask]), 0)
    assert np.isclose(np.std(processed_data[processed_mask]), 1)


@pytest.mark.parametrize("input_idxs, output_idxs", [[(0, 2), (0, 5)], [(1, 3), (1, 2, 3, 4, 6, 7, 8)]])
def test_process_data_subset_by_group(data_processor, input_idxs, output_idxs):
    # In this test, each group has num_unproc_col = 1, hence we can subset using input_idxs
    processed_subset = data_processor.process_data_subset_by_group(good_data[:, input_idxs], np.array(input_idxs))
    good_subset = processed_good_data[:, output_idxs]
    print(processed_subset, good_subset)
    assert np.allclose(processed_subset, good_subset)


def test_process_mask(data_processor):
    processed_mask = data_processor.process_mask(good_mask)
    assert np.all(processed_mask == processed_good_mask)


def test_revert_data(data_processor):
    reverted_data = data_processor.revert_data(processed_good_data)
    assert np.all(np.isclose(reverted_data.astype(float), good_data))


def test_revert_data_not_squashed(data_processor_not_squashed):
    reverted_data = data_processor_not_squashed.revert_data(processed_good_data_not_squashed)
    assert np.all(np.isclose(reverted_data.astype(float), good_data))


def test_revert_data_standardize(data_processor_standardize):
    processed_data = data_processor_standardize.process_data(good_data)
    reverted_data = data_processor_standardize.revert_data(processed_data)
    assert np.all(np.isclose(reverted_data.astype(float), good_data))


def test_revert_mask(data_processor):
    reverted_mask = data_processor.revert_mask(processed_good_mask)
    assert np.all(reverted_mask == good_mask)


def test_process_data_and_masks(data_processor):
    processed_data, processed_mask = data_processor.process_data_and_masks(good_data, good_mask)
    assert np.all(np.isclose(processed_data, processed_good_data))
    assert np.all(processed_mask == processed_good_mask)


def test_process_data_and_masks_two_masks(data_processor):
    (
        processed_data,
        processed_mask,
        processed_extra_mask,
    ) = data_processor.process_data_and_masks(good_data, good_mask, good_mask)
    assert np.all(np.isclose(processed_data, processed_good_data))
    assert np.all(processed_mask == processed_good_mask)
    assert np.all(processed_extra_mask == processed_good_mask)


def test_process_data_and_masks_sparse(data_processor):
    sparse_good_data = csr_matrix(good_data)
    sparse_good_mask = csr_matrix(good_mask)
    processed_data, processed_mask = data_processor.process_data_and_masks(sparse_good_data, sparse_good_mask)
    assert np.all(np.isclose(processed_data.toarray(), processed_good_data))
    assert np.all(processed_mask.toarray() == processed_good_mask)


def test_process_data_and_masks_two_masks_sparse(data_processor):
    sparse_good_data = csr_matrix(good_data)
    sparse_good_mask = csr_matrix(good_mask)
    (
        processed_data,
        processed_mask,
        processed_extra_mask,
    ) = data_processor.process_data_and_masks(sparse_good_data, sparse_good_mask, sparse_good_mask)
    assert np.all(np.isclose(processed_data.toarray(), processed_good_data))
    assert np.all(processed_mask.toarray() == processed_good_mask)
    assert np.all(processed_extra_mask.toarray() == processed_good_mask)


def test_split_contiguous_sublistss(data_processor):
    assert data_processor.split_contiguous_sublists([1, 2, 4, 6, 7]) == [
        [1, 2],
        [4],
        [6, 7],
    ]
    assert data_processor.split_contiguous_sublists([]) == []
    assert data_processor.split_contiguous_sublists([1]) == [[1]]
