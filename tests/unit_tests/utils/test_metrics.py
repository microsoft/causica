import numpy as np
import pytest
from scipy.sparse import csr_matrix

from causica.datasets.variables import Variable, Variables
from causica.utils.metrics import (
    get_aggregated_accuracy,
    get_aggregated_binary_confusion_matrix,
    get_area_under_ROC_PR,
    get_fraction_incorrectly_classified,
    get_rmse,
)


@pytest.mark.parametrize("sparse_mask", [False, True])
def test_fraction_incorrect_binary_var(sparse_mask):
    imputed_values = np.expand_dims(np.array([0, 1, 0]), axis=1)
    ground_truth = np.expand_dims(np.array([1, 1, 0]), axis=1)
    target_mask = np.expand_dims(np.array([1, 1, 1], dtype=bool), axis=1)
    if sparse_mask:
        target_mask = csr_matrix(target_mask)

    frac_incorrect = get_fraction_incorrectly_classified(imputed_values, ground_truth, target_mask)
    assert np.isclose(frac_incorrect, 1 / 3)


@pytest.mark.parametrize("sparse_mask", [False, True])
def test_fraction_incorrect_binary_var_missing_data(sparse_mask):
    imputed_values = np.expand_dims(np.array([0, 1, 0]), axis=1)
    ground_truth = np.expand_dims(np.array([1, 1, 0]), axis=1)
    target_mask = np.expand_dims(np.array([1, 1, 0], dtype=bool), axis=1)
    if sparse_mask:
        target_mask = csr_matrix(target_mask)

    frac_incorrect = get_fraction_incorrectly_classified(imputed_values, ground_truth, target_mask)
    assert np.isclose(frac_incorrect, 1 / 2)


@pytest.mark.parametrize("sparse_mask", [False, True])
def test_fraction_incorrect_categorical_var(sparse_mask):
    imputed_values = np.expand_dims(np.array([0, 2, 3]), axis=1)
    ground_truth = np.expand_dims(np.array([1, 2, 3]), axis=1)
    target_mask = np.expand_dims(np.array([1, 1, 1], dtype=bool), axis=1)
    if sparse_mask:
        target_mask = csr_matrix(target_mask)

    frac_incorrect = get_fraction_incorrectly_classified(imputed_values, ground_truth, target_mask)
    assert np.isclose(frac_incorrect, 1 / 3)


@pytest.mark.parametrize("sparse_mask", [False, True])
def test_fraction_incorrect_categorical_var_missing_data(sparse_mask):
    imputed_values = np.expand_dims(np.array([0, 2, 3]), axis=1)
    ground_truth = np.expand_dims(np.array([1, 2, 3]), axis=1)
    target_mask = csr_matrix(np.expand_dims(np.array([1, 1, 0], dtype=bool), axis=1))
    if sparse_mask:
        target_mask = csr_matrix(target_mask)

    frac_incorrect = get_fraction_incorrectly_classified(imputed_values, ground_truth, target_mask)
    assert np.isclose(frac_incorrect, 1 / 2)


@pytest.mark.parametrize("sparse_mask", [False, True])
def test_rmse_for_binary_variable(sparse_mask):
    imputed_values = np.expand_dims(np.array([1.0, 0.0, 0.0]), axis=1)
    ground_truth = np.expand_dims(np.array([1.0, 1.0, 1.0]), axis=1)
    target_mask = np.expand_dims(np.array([1, 1, 1], dtype=bool), axis=1)
    if sparse_mask:
        target_mask = csr_matrix(target_mask)
    variables = Variables([Variable("binary_input", True, "binary", 0, 1)])

    rmse_val = get_rmse(imputed_values, ground_truth, target_mask, variables, False)
    assert np.isclose(rmse_val, np.sqrt(2 / 3))


@pytest.mark.parametrize("sparse_mask", [False, True])
def test_rmse_for_continuous_variable_with_mask(sparse_mask):
    imputed_values = np.expand_dims(np.array([1.0, 3.0, 2.0]), axis=1)
    ground_truth = np.expand_dims(np.array([3.0, 1.0, 1.0]), axis=1)
    target_mask = np.expand_dims(np.array([1, 0, 0], dtype=bool), axis=1)
    if sparse_mask:
        target_mask = csr_matrix(target_mask)
    variables = Variables(
        [
            Variable(
                "continuous_input",
                True,
                "continuous",
                1.0,
                3.0,
            )
        ]
    )

    rmse_val = get_rmse(imputed_values, ground_truth, target_mask, variables, False)
    assert rmse_val == pytest.approx(2, 0.1)


@pytest.mark.parametrize("sparse_mask", [False, True])
def test_rmse_for_categorical_variable(sparse_mask):
    imputed_values = np.expand_dims(np.array([1.0, 3.0, 3.0]), axis=1)
    ground_truth = np.expand_dims(np.array([3.0, 3.0, 1.0]), axis=1)
    target_mask = np.expand_dims(np.array([1, 1, 1], dtype=bool), axis=1)
    if sparse_mask:
        target_mask = csr_matrix(target_mask)
    variables = Variables([Variable("categorical_input", True, "categorical", 1.0, 3.0)])

    rmse_val = get_rmse(imputed_values, ground_truth, target_mask, variables, False)
    assert np.isclose(rmse_val, np.sqrt(2 / 3))


@pytest.mark.parametrize("sparse_mask", [False, True])
def test_rmse_for_mixed_type(sparse_mask):
    imputed_values = np.array(
        [
            [3.2, 1, 0.7],
            [6.4, 2, 0.4],
            [-0.3, 3, 0.3],
        ]
    )
    ground_truth = np.array(
        [
            [3, 3, 1],
            [6, 2, 0],
            [0, 1, 1],
        ]
    )
    target_mask = np.ones((3, 3), dtype=bool)
    if sparse_mask:
        target_mask = csr_matrix(target_mask)
    variables = Variables(
        [
            Variable("continuous_input", True, "continuous", -10, 10),
            Variable("categorical_input", True, "categorical", 1.0, 3.0),
            Variable("binary_input", True, "binary", 0, 1),
        ]
    )

    rmse_val = get_rmse(imputed_values, ground_truth, target_mask, variables, False)
    expected_val = np.sqrt((0.2**2 + 0.4**2 + 0.3**2 + 3) / 9)
    assert np.isclose(rmse_val, expected_val)


@pytest.mark.parametrize("sparse_mask", [False, True])
def test_rmse_batching(sparse_mask):
    # Repeat all rows 10 times
    imputed_values = np.tile(np.expand_dims(np.array([1.0, 0.0, 0.0]), axis=1), (1, 10))
    ground_truth = np.tile(np.expand_dims(np.array([1.0, 1.0, 1.0]), axis=1), (1, 10))
    target_mask = np.tile(np.expand_dims(np.array([1, 1, 1], dtype=bool), axis=1), (1, 10))
    if sparse_mask:
        target_mask = csr_matrix(target_mask)
    variables = Variables([Variable("binary_input", True, "binary", 0, 1)])

    # Choose batch size that doesn't divide number of rows to ensure last batch handled correctly.
    rmse_val = get_rmse(imputed_values, ground_truth, target_mask, variables, False, batch_size=7)
    assert np.isclose(rmse_val, np.sqrt(2 / 3))


@pytest.mark.parametrize("sparse_mask", [False, True])
def test_aggregated_accuracy_return_val(sparse_mask):
    user_count = 3
    feature_count = 2
    imputed_values = np.ones((user_count, feature_count))
    ground_truth = np.ones((user_count, feature_count))
    target_mask = np.ones((user_count, feature_count), dtype=bool)
    if sparse_mask:
        target_mask = csr_matrix(target_mask)
    variables = Variables([Variable("binary_input", True, "binary"), Variable("binary_input_2", True, "binary")])

    output = get_aggregated_accuracy(imputed_values, ground_truth, target_mask, variables)

    assert isinstance(output, float)


@pytest.mark.parametrize("sparse_mask", [False, True])
def test_confusion_matrix(sparse_mask):
    imputed_values = np.expand_dims(np.array([0, 0, 1]), axis=1)
    ground_truth = np.expand_dims(np.array([0, 1, 1]), axis=1)
    target_mask = np.expand_dims(np.array([1, 1, 1], dtype=bool), axis=1)
    if sparse_mask:
        target_mask = csr_matrix(target_mask)
    variables = Variables([Variable("binary_input", True, "binary", 0, 1)])

    cm = get_aggregated_binary_confusion_matrix(imputed_values, ground_truth, target_mask, variables)
    expected = np.array([[1.0, 0.5], [0.0, 0.5]])
    assert np.all(cm == expected)


@pytest.mark.parametrize("sparse_mask", [False, True])
def test_area_under_ROC_PR(sparse_mask):
    ground_truth = np.expand_dims(np.array([0, 0, 1, 1]), axis=1)
    target_mask = np.expand_dims(np.array([1, 1, 1, 1], dtype=bool), axis=1)
    if sparse_mask:
        target_mask = csr_matrix(target_mask)

    variables = Variables([Variable("binary_input", True, "binary", 0, 1)])
    imputed_values = np.expand_dims(np.array([0, 0, 1, 1]), axis=1)

    AUROC, AUPR = get_area_under_ROC_PR(imputed_values, ground_truth, target_mask, variables)
    assert (AUROC == 1) & (AUPR == 1)
