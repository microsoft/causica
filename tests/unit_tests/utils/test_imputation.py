import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from causica.datasets.dataset import Dataset, SparseDataset
from causica.datasets.variables import Variable, Variables
from causica.utils.imputation import eval_imputation, split_mask

from .mock_model_for_objective import MockModelForObjective

input_dim = 8  # count in the 5-class categorical variable and 3 other variables
output_dim = 8
set_embedding_dim = 4
embedding_dim = 2
latent_dim = 17


# pylint: disable=redefined-outer-name
@pytest.fixture(scope="function")
def variables():
    return Variables(
        [
            Variable("binary_input", True, "binary", 0.0, 1.0),
            Variable("numeric_input", True, "continuous", 3, 13),
            Variable("categorical_input", True, "categorical", 1, 5),
            Variable("numeric_target", False, "continuous", 2, 300),
        ]
    )


def test_eval_imputation_row_split(variables, tmpdir):
    model = MockModelForObjective("mock_model", variables, tmpdir)
    data = np.tile(np.array([0.0, 4.6, 3.0, 80.0])[np.newaxis, :], (5, 1))
    mask = np.ones_like(data, dtype=bool)
    dataset = Dataset(data, mask, data, mask, data, mask, variables)
    impute_config = {"sample_count": 5}
    (
        train_obs_mask,
        train_target_mask,
        train_imputations,
        train_metrics,
        val_obs_mask,
        val_target_mask,
        val_imputations,
        val_metrics,
        test_obs_mask,
        test_target_mask,
        test_imputations,
        test_metrics,
    ) = eval_imputation(dataset, model, variables, "rows", None, impute_config, True, 0)

    assert isinstance(train_obs_mask, np.ndarray)
    assert isinstance(train_target_mask, np.ndarray)
    assert isinstance(train_imputations, np.ndarray)
    assert isinstance(train_metrics, dict)
    assert isinstance(val_obs_mask, np.ndarray)
    assert isinstance(val_target_mask, np.ndarray)
    assert isinstance(val_imputations, np.ndarray)
    assert isinstance(val_metrics, dict)
    assert isinstance(test_obs_mask, np.ndarray)
    assert isinstance(test_target_mask, np.ndarray)
    assert isinstance(test_imputations, np.ndarray)
    assert isinstance(test_metrics, dict)


def test_eval_imputation_row_split_no_val_data(variables, tmpdir):
    model = MockModelForObjective("mock_model", variables, tmpdir)
    data = np.tile(np.array([0.0, 4.6, 3.0, 80.0])[np.newaxis, :], (5, 1))
    mask = np.ones_like(data, dtype=bool)
    dataset = Dataset(data, mask, None, None, data, mask, variables)
    impute_config = {"sample_count": 5}
    (
        train_obs_mask,
        train_target_mask,
        train_imputations,
        train_metrics,
        val_obs_mask,
        val_target_mask,
        val_imputations,
        val_metrics,
        test_obs_mask,
        test_target_mask,
        test_imputations,
        test_metrics,
    ) = eval_imputation(dataset, model, variables, "rows", None, impute_config, True, 0)

    assert isinstance(train_obs_mask, np.ndarray)
    assert isinstance(train_target_mask, np.ndarray)
    assert isinstance(train_imputations, np.ndarray)
    assert isinstance(train_metrics, dict)
    assert val_obs_mask is None
    assert val_target_mask is None
    assert val_imputations is None
    assert not val_metrics
    assert isinstance(test_obs_mask, np.ndarray)
    assert isinstance(test_target_mask, np.ndarray)
    assert isinstance(test_imputations, np.ndarray)
    assert isinstance(test_metrics, dict)


def test_eval_imputation_row_split_no_train_imputation(variables, tmpdir):
    model = MockModelForObjective("mock_model", variables, tmpdir)
    data = np.tile(np.array([0.0, 4.6, 3.0, 80.0])[np.newaxis, :], (5, 1))
    mask = np.ones_like(data, dtype=bool)
    dataset = Dataset(data, mask, data, mask, data, mask, variables)
    impute_config = {"sample_count": 5}
    (
        train_obs_mask,
        train_target_mask,
        train_imputations,
        train_metrics,
        val_obs_mask,
        val_target_mask,
        val_imputations,
        val_metrics,
        test_obs_mask,
        test_target_mask,
        test_imputations,
        test_metrics,
    ) = eval_imputation(dataset, model, variables, "rows", None, impute_config, False, 0)

    assert train_obs_mask is None
    assert train_target_mask is None
    assert train_imputations is None
    assert not train_metrics
    assert isinstance(val_obs_mask, np.ndarray)
    assert isinstance(val_target_mask, np.ndarray)
    assert isinstance(val_imputations, np.ndarray)
    assert isinstance(val_metrics, dict)
    assert isinstance(test_obs_mask, np.ndarray)
    assert isinstance(test_target_mask, np.ndarray)
    assert isinstance(test_imputations, np.ndarray)
    assert isinstance(test_metrics, dict)


@pytest.mark.parametrize("sparse", [False, True])
def test_eval_imputation_element_split(variables, tmpdir, sparse):
    model = MockModelForObjective("mock_model", variables, tmpdir)
    data = np.tile(np.array([0.0, 4.6, 3.0, 80.0])[np.newaxis, :], (5, 1))
    # Split by columns rather than rows as an example elementwise data split
    mask = np.zeros_like(data, dtype=bool)
    if sparse:
        data = csr_matrix(data)
        mask = csr_matrix(mask)
    train_mask = mask.copy()
    train_mask[:, 0:2] = 1
    if sparse:
        train_data = data.multiply(train_mask)
    else:
        train_data = data * train_mask
    val_mask = mask.copy()
    val_mask[:, 2] = 1
    if sparse:
        val_data = data.multiply(val_mask)
    else:
        val_data = data * val_mask
    test_mask = mask.copy()
    test_mask[:, 3] = 1
    if sparse:
        test_data = data.multiply(test_mask)
    else:
        test_data = data * test_mask
    if sparse:
        dataset = SparseDataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables)
    else:
        dataset = Dataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables)
    impute_config = {"sample_count": 5}
    (
        train_obs_mask,
        train_target_mask,
        train_imputations,
        train_metrics,
        val_obs_mask,
        val_target_mask,
        val_imputations,
        val_metrics,
        test_obs_mask,
        test_target_mask,
        test_imputations,
        test_metrics,
    ) = eval_imputation(dataset, model, variables, "elements", None, impute_config, True, 0)

    assert train_obs_mask is None
    assert train_target_mask is None
    assert train_imputations is None
    assert not train_metrics
    if sparse:
        assert issparse(val_obs_mask)
        assert issparse(val_target_mask)
        assert np.all(val_obs_mask.toarray() == train_mask.toarray())
        assert np.all(val_target_mask.toarray() == val_mask.toarray())
    else:
        assert np.all(val_obs_mask == train_mask)
        assert np.all(val_target_mask == val_mask)
    assert isinstance(val_imputations, np.ndarray)
    assert isinstance(val_metrics, dict)
    if sparse:
        assert issparse(test_obs_mask)
        assert issparse(test_target_mask)
        assert np.all(test_obs_mask.toarray() == train_mask.toarray())
        assert np.all(test_target_mask.toarray() == test_mask.toarray())
    else:
        assert np.all(test_obs_mask == train_mask)
        assert np.all(test_target_mask == test_mask)
    assert isinstance(test_imputations, np.ndarray)
    assert isinstance(test_metrics, dict)


@pytest.mark.parametrize("sparse", [False, True])
def test_eval_imputation_element_split_rows_without_targets(variables, tmpdir, sparse):
    model = MockModelForObjective("mock_model", variables, tmpdir)
    data = np.tile(np.array([0.0, 4.6, 3.0, 80.0])[np.newaxis, :], (5, 1))
    # Split by columns rather than rows as an example elementwise data split
    mask = np.zeros_like(data, dtype=bool)
    if sparse:
        data = csr_matrix(data)
        mask = csr_matrix(mask)
    train_mask = mask.copy()
    train_mask[:, 0:2] = 1
    if sparse:
        train_data = data.multiply(train_mask)
    else:
        train_data = data * train_mask
    val_mask = mask.copy()
    val_mask[2:3, 2] = 1
    if sparse:
        val_data = data.multiply(val_mask)
    else:
        val_data = data * val_mask
    test_mask = mask.copy()
    test_mask[0:2, 3] = 1
    if sparse:
        test_data = data.multiply(test_mask)
    else:
        test_data = data * test_mask
    if sparse:
        dataset = SparseDataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables)
    else:
        dataset = Dataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables)
    impute_config = {"sample_count": 5}
    (
        train_obs_mask,
        train_target_mask,
        train_imputations,
        train_metrics,
        val_obs_mask,
        val_target_mask,
        val_imputations,
        val_metrics,
        test_obs_mask,
        test_target_mask,
        test_imputations,
        test_metrics,
    ) = eval_imputation(dataset, model, variables, "elements", None, impute_config, True, 0)

    assert train_obs_mask is None
    assert train_target_mask is None
    assert train_imputations is None
    assert not train_metrics
    if sparse:
        assert issparse(val_obs_mask)
        assert issparse(val_target_mask)
        assert np.all(val_obs_mask.toarray() == train_mask.toarray())
        assert np.all(val_target_mask.toarray() == val_mask.toarray())
    else:
        assert np.all(val_obs_mask == train_mask)
        assert np.all(val_target_mask == val_mask)
    assert isinstance(val_imputations, np.ndarray)
    assert isinstance(val_metrics, dict)
    if sparse:
        assert issparse(test_obs_mask)
        assert issparse(test_target_mask)
        assert np.all(test_obs_mask.toarray() == train_mask.toarray())
        assert np.all(test_target_mask.toarray() == test_mask.toarray())
    else:
        assert np.all(test_obs_mask == train_mask)
        assert np.all(test_target_mask == test_mask)
    assert isinstance(test_imputations, np.ndarray)
    assert isinstance(test_metrics, dict)
    assert test_imputations.shape == (sum(np.ravel(np.sum(test_mask + val_mask, 1)) != 0), mask.shape[1])


@pytest.mark.parametrize("sparse", [False, True])
def test_eval_imputation_element_split_no_val_data(variables, tmpdir, sparse):
    model = MockModelForObjective("mock_model", variables, tmpdir)
    data = np.tile(np.array([0.0, 4.6, 3.0, 80.0])[np.newaxis, :], (5, 1))
    # Split by columns rather than rows as an example elementwise data split
    mask = np.zeros_like(data, dtype=bool)
    if sparse:
        data = csr_matrix(data)
        mask = csr_matrix(mask)
    train_mask = mask.copy()
    train_mask[:, 0:2] = 1
    if sparse:
        train_data = data.multiply(train_mask)
    else:
        train_data = data * train_mask
    val_data = None
    val_mask = None
    test_mask = mask.copy()
    test_mask[:, 3:] = 1
    if sparse:
        test_data = data.multiply(test_mask)
    else:
        test_data = data * test_mask
    if sparse:
        dataset = SparseDataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables)
    else:
        dataset = Dataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables)
    impute_config = {"sample_count": 5}
    (
        train_obs_mask,
        train_target_mask,
        train_imputations,
        train_metrics,
        val_obs_mask,
        val_target_mask,
        val_imputations,
        val_metrics,
        test_obs_mask,
        test_target_mask,
        test_imputations,
        test_metrics,
    ) = eval_imputation(dataset, model, variables, "elements", None, impute_config, True, 0)

    assert train_obs_mask is None
    assert train_target_mask is None
    assert train_imputations is None
    assert not train_metrics
    assert val_obs_mask is None
    assert val_target_mask is None
    assert val_imputations is None
    assert not val_metrics
    if sparse:
        assert issparse(test_obs_mask)
        assert issparse(test_target_mask)
        assert np.all(test_obs_mask.toarray() == train_mask.toarray())
        assert np.all(test_target_mask.toarray() == test_mask.toarray())
    else:
        assert np.all(test_obs_mask == train_mask)
        assert np.all(test_target_mask == test_mask)
    assert isinstance(test_imputations, np.ndarray)
    assert isinstance(test_metrics, dict)


@pytest.mark.parametrize("sparse", [False, True])
def test_eval_imputation_element_split_no_val_data_rows_without_target(variables, tmpdir, sparse):
    model = MockModelForObjective("mock_model", variables, tmpdir)
    data = np.tile(np.array([0.0, 4.6, 3.0, 80.0])[np.newaxis, :], (5, 1))
    # Split by columns rather than rows as an example elementwise data split
    mask = np.zeros_like(data, dtype=bool)
    if sparse:
        data = csr_matrix(data)
        mask = csr_matrix(mask)
    train_mask = mask.copy()
    train_mask[:, 0:2] = 1
    if sparse:
        train_data = data.multiply(train_mask)
    else:
        train_data = data * train_mask
    val_data = None
    val_mask = None
    test_mask = mask.copy()
    test_mask[0:2, 3:] = 1
    if sparse:
        test_data = data.multiply(test_mask)
    else:
        test_data = data * test_mask
    if sparse:
        dataset = SparseDataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables)
    else:
        dataset = Dataset(train_data, train_mask, val_data, val_mask, test_data, test_mask, variables)
    impute_config = {"sample_count": 5}
    (
        train_obs_mask,
        train_target_mask,
        train_imputations,
        train_metrics,
        val_obs_mask,
        val_target_mask,
        val_imputations,
        val_metrics,
        test_obs_mask,
        test_target_mask,
        test_imputations,
        test_metrics,
    ) = eval_imputation(dataset, model, variables, "elements", None, impute_config, True, 0)

    assert train_obs_mask is None
    assert train_target_mask is None
    assert train_imputations is None
    assert not train_metrics
    assert val_obs_mask is None
    assert val_target_mask is None
    assert val_imputations is None
    assert not val_metrics
    if sparse:
        assert issparse(test_obs_mask)
        assert issparse(test_target_mask)
        assert np.all(test_obs_mask.toarray() == train_mask.toarray())
        assert np.all(test_target_mask.toarray() == test_mask.toarray())
    else:
        assert np.all(test_obs_mask == train_mask)
        assert np.all(test_target_mask == test_mask)
    assert isinstance(test_imputations, np.ndarray)
    assert isinstance(test_metrics, dict)
    assert test_imputations.shape == (sum(np.ravel(np.sum(test_mask, 1)) != 0), mask.shape[1])


def test_split_mask():
    mask = np.arange(100).reshape(10, 10) % 2 == 0  # Mask alternating 0 and 1
    observed_mask, target_mask = split_mask(mask)
    assert observed_mask.shape == mask.shape
    assert observed_mask.shape == target_mask.shape
    assert np.all(observed_mask + target_mask == mask)
    assert (observed_mask + target_mask).max() <= 1  # Check that split masks do not intersect.


def test_split_mask_sparse():
    mask = np.arange(100).reshape(10, 10) % 2 == 0  # Mask alternating 0 and 1
    mask = csr_matrix(mask)
    observed_mask, target_mask = split_mask(mask)
    assert issparse(observed_mask)
    assert issparse(target_mask)
    assert observed_mask.shape == mask.shape
    assert observed_mask.shape == target_mask.shape
    assert ((observed_mask + target_mask) - mask).nnz == 0  # If obs + target = original then no non-zero elements here.
    assert (observed_mask + target_mask).max() <= 1  # Check that split masks do not intersect.


def test_split_mask_proportion():
    target_prob = 0.3
    n_rows = 10
    n_features = 10000000
    mask = np.ones((n_rows, n_features), dtype=bool)
    observed_mask, target_mask = split_mask(mask, target_prob, seed=42)
    assert np.mean(observed_mask) == pytest.approx(1 - target_prob, rel=1e-4, abs=1e-12)
    assert np.mean(target_mask) == pytest.approx(target_prob, rel=1e-4, abs=1e-12)


def test_split_mask_proportion_missing_values():
    target_prob = 0.3
    n_rows = 10
    n_features = 10000000
    mask = np.arange(n_rows * n_features).reshape(n_rows, n_features) % 2 == 0  # Mask alternates 0 and 1
    observed_mask, target_mask = split_mask(mask, target_prob, seed=42)
    # Divide expected numbers of 1s in each mask by 2 since original mask is only observed for 50% of values, and
    # split mask should only return 1s in each mask where there were observations in the original mask.
    assert np.mean(observed_mask) == pytest.approx((1 - target_prob) / 2, rel=1e-3, abs=1e-12)
    assert np.mean(target_mask) == pytest.approx(target_prob / 2, rel=1e-3, abs=1e-12)


def test_split_mask_seed_deterministic():
    mask = np.arange(100).reshape(10, 10) % 2 == 0  # Mask alternating 0 and 1
    observed_mask_1, target_mask_1 = split_mask(mask, seed=0)
    observed_mask_2, target_mask_2 = split_mask(mask, seed=0)
    assert np.array_equal(
        observed_mask_1, observed_mask_2
    )  # Check that setting the same seed results in the same observed and target masks
    assert np.array_equal(target_mask_1, target_mask_2)

    observed_mask_1, target_mask_1 = split_mask(mask, seed=0)
    observed_mask_2, target_mask_2 = split_mask(mask, seed=1)
    assert not np.array_equal(
        observed_mask_1, observed_mask_2
    )  # Check that setting a different seed results in different observed and target masks
    assert not np.array_equal(target_mask_1, target_mask_2)
