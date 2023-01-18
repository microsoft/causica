import math

import numpy as np
import pytest
import torch

from causica.datasets.variables import Variable, Variables
from causica.models.deci.base_distributions import BinaryLikelihood, CategoricalLikelihood
from causica.models.deci.deci import DECI
from causica.utils.causality_utils import process_adjacency_mats

five_cts_ungrouped_variables = Variables(
    [
        Variable("continuous_input_1", True, "continuous", 0, 1),
        Variable("continuous_input_2", True, "continuous", 0, 1),
        Variable("continuous_input_3", True, "continuous", 0, 1),
        Variable("continuous_input_4", True, "continuous", 0, 1),
        Variable("continuous_input_5", True, "continuous", 0, 1),
    ]
)


six_cts_grouped_variables = Variables(
    [
        Variable("a", True, "continuous", 0, 1, group_name="Group 1"),
        Variable("b", True, "continuous", 0, 1, group_name="Group 1"),
        Variable("c", True, "continuous", 0, 1, group_name="Group 2"),
        Variable("d", True, "continuous", 0, 1, group_name="Group 2"),
        Variable("e", True, "continuous", 0, 1, group_name="Group 3"),
        Variable("f", True, "continuous", 0, 1, group_name="Group 3"),
    ]
)


cts_and_discrete_variables = Variables(
    [
        Variable("a", True, "continuous", 0, 1, group_name="Group 1"),
        Variable("b", True, "continuous", 0, 1, group_name="Group 1"),
        Variable("c", True, "categorical", 0, 3),
        Variable("d", True, "categorical", 0, 3),
        Variable("e", True, "categorical", 0, 3),
        Variable("f", True, "binary", 0, 1),
    ]
)


mixed_type_group = Variables(
    [
        Variable("a", True, "continuous", 0, 1, group_name="Group 1"),
        Variable("b", True, "binary", 0, 1, group_name="Group 1"),
        Variable("c", True, "continuous", 0, 1, group_name="Group 2"),
        Variable("d", True, "continuous", 0, 1, group_name="Group 2"),
    ]
)


two_cat_one_group = Variables(
    [
        Variable("a", True, "categorical", 0, 1, group_name="Group 1"),
        Variable("b", True, "categorical", 0, 1, group_name="Group 1"),
        Variable("c", True, "continuous", 0, 1, group_name="Group 2"),
        Variable("d", True, "continuous", 0, 1, group_name="Group 2"),
    ]
)


@pytest.fixture
def model_config():
    return {
        "tau_gumbel": 1.0,
        "lambda_dag": 100.0,
        "lambda_sparse": 1.0,
        "base_distribution_type": "gaussian",
        "var_dist_A_mode": "enco",
        "imputer_layer_sizes": [20],
        "random_seed": [0],
    }


# pylint: disable=redefined-outer-name
@pytest.mark.parametrize(
    "variables",
    [
        five_cts_ungrouped_variables,
        six_cts_grouped_variables,
        cts_and_discrete_variables,
        mixed_type_group,
        two_cat_one_group,
    ],
)
def test_dagness_factor(tmpdir_factory, model_config, variables):
    model = DECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    d = variables.num_groups

    A = torch.triu(torch.rand(d, d)) * (1 - torch.eye(d))
    A = A.round()

    assert model.dagness_factor(A) == 0


@pytest.mark.parametrize("variables", [five_cts_ungrouped_variables, six_cts_grouped_variables])
def test_imputation(tmpdir_factory, model_config, variables):

    model = DECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    d = variables.num_processed_cols
    n = 100

    data = np.random.rand(n, d)
    mask = np.random.rand(n, len(variables)).round()

    data_reconstructed = model.impute(data, mask, average=True)

    assert data_reconstructed.shape == data.shape

    impute_config_dict = {"sample_count": 10}

    data_reconstructed = model.impute(data, mask, impute_config_dict=impute_config_dict, average=False)

    assert data_reconstructed.mean(axis=0).shape == data.shape


@pytest.mark.parametrize(
    "variables",
    [
        five_cts_ungrouped_variables,
        six_cts_grouped_variables,
        cts_and_discrete_variables,
        mixed_type_group,
        two_cat_one_group,
    ],
)
def test_sample(tmpdir_factory, model_config, variables):

    model = DECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    d = variables.num_processed_cols

    base_samples = model.sample(Nsamples=100, most_likely_graph=False)
    assert base_samples.shape == (100, d)
    # Test valid one-hot categorical samples and binary samples
    for region, variable in zip(variables.processed_cols, variables):
        if variable.type_ == "categorical":
            assert (base_samples[:, region].max(-1)[0] == 1).all()
            assert (base_samples[:, region].min(-1)[0] == 0).all()
            assert (base_samples[:, region].sum(-1) == 1).all()
        elif variable.type_ == "binary":
            assert (base_samples[:, region].max(-1)[0] <= 1).all()
            assert (base_samples[:, region].min(-1)[0] >= 0).all()

    # Intervene on node 0
    intervention_idxs = torch.tensor([0])
    # This will broadcast as appropriate for vector-valued nodes
    intervention_values = torch.tensor([0.0])

    intervention_samples = model.sample(
        Nsamples=100,
        most_likely_graph=True,
        samples_per_graph=100,
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
    )

    var_0_mask = model.ICGNN.group_mask[0, :]
    var_1_mask = model.ICGNN.group_mask[1, :]

    assert intervention_samples.shape == (100, d)
    assert torch.all(intervention_samples[:, var_0_mask] == 0)
    assert not torch.all(intervention_samples[:, var_1_mask] == 0)


def test_sample_discrete_intervention(tmpdir_factory, model_config):

    model = DECI.create(
        model_id="model_id",
        variables=cts_and_discrete_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    d = cts_and_discrete_variables.num_processed_cols

    intervention_idxs = torch.tensor([1])
    intervention_values = torch.tensor([0, 0, 1, 0])

    intervention_samples = model.sample(
        Nsamples=100,
        most_likely_graph=True,
        samples_per_graph=100,
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
    )

    assert np.allclose(intervention_samples[0, 2:6], np.array([0, 0, 1, 0]))
    assert intervention_samples.shape == (100, d)


def test_sample_multi_intervention(tmpdir_factory, model_config):

    model = DECI.create(
        model_id="model_id",
        variables=cts_and_discrete_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    d = cts_and_discrete_variables.num_processed_cols

    # Variable 0 is a continuous variable of dimension 2
    # Variable 1 is a categorical variable on 4 categories
    intervention_idxs = torch.tensor([0, 1])
    intervention_values = torch.tensor([0.5, 0.5, 0, 0, 1, 0])

    intervention_samples = model.sample(
        Nsamples=100,
        most_likely_graph=True,
        samples_per_graph=100,
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
    )

    assert np.allclose(intervention_samples[0, :6], np.array([0.5, 0.5, 0, 0, 1, 0]))
    assert intervention_samples.shape == (100, d)


@pytest.mark.parametrize(
    "variables",
    [
        five_cts_ungrouped_variables,
        six_cts_grouped_variables,
        cts_and_discrete_variables,
        mixed_type_group,
        two_cat_one_group,
    ],
)
def test_log_prob(tmpdir_factory, model_config, variables):

    model = DECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    d = variables.num_processed_cols

    X = torch.rand(10, d)
    for region, variable in zip(variables.processed_cols, variables):
        if variable.type_ == "categorical":
            maxes = X[:, region].max(-1, keepdim=True)[0]
            X[:, region] = (X[:, region] >= maxes).float()
        if variable.type_ == "binary":
            X[:, region] = (X[:, region] > 0.5).float()

    deterministic_log_p0 = model.log_prob(X, Nsamples_per_graph=100, most_likely_graph=True)

    deterministic_log_p1 = model.log_prob(X, Nsamples_per_graph=1, most_likely_graph=True)

    assert np.allclose(deterministic_log_p1, deterministic_log_p0)

    # This just tests that the stochastic mode runs
    _ = model.log_prob(X, Nsamples_per_graph=10, most_likely_graph=False)

    intervention_idxs = torch.tensor([0])
    intervention_values = torch.tensor([0.0])

    deterministic_intervention_log_p = model.log_prob(
        X, most_likely_graph=True, intervention_idxs=intervention_idxs, intervention_values=intervention_values
    )

    assert not np.allclose(deterministic_log_p1, deterministic_intervention_log_p)


# TODO: make ATE and CATE tests more comprehensive. Currently they only check that the method does not throw exceptions


@pytest.mark.parametrize(
    "variables",
    [
        five_cts_ungrouped_variables,
        six_cts_grouped_variables,
        cts_and_discrete_variables,
        mixed_type_group,
        two_cat_one_group,
    ],
)
def test_ate(tmpdir_factory, model_config, variables):

    model = DECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    d = variables.num_processed_cols

    intervention_idxs = torch.tensor([0])
    intervention_values = torch.tensor([0.0])

    cate0, cate0_norm = model.cate(
        intervention_idxs,
        intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        Nsamples_per_graph=2,
        Ngraphs=1000,
        most_likely_graph=False,
    )

    cate1, cate1_norm = model.cate(
        intervention_idxs,
        intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        Nsamples_per_graph=2,
        Ngraphs=1000,
        most_likely_graph=False,
    )

    #  Note that graph is deterministic here but samples are still random
    cate_det0, cate_det0_norm = model.cate(
        intervention_idxs,
        intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        Nsamples_per_graph=2000,
        Ngraphs=1,
        most_likely_graph=True,
    )

    cate_det1, cate_det1_norm = model.cate(
        intervention_idxs,
        intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        Nsamples_per_graph=2000,
        Ngraphs=1,
        most_likely_graph=True,
    )

    assert cate0.shape == (d,)
    assert cate0_norm.shape == (d,)

    # Test values for continuous only
    assert np.all((cate0 - cate1)[variables.continuous_idxs] != 0)
    assert np.all((cate0_norm - cate1_norm)[variables.continuous_idxs] != 0)

    assert np.all((cate_det0 - cate_det1)[variables.continuous_idxs] != 0)
    assert np.all((cate_det0_norm - cate_det1_norm)[variables.continuous_idxs] != 0)

    with pytest.raises(AssertionError, match="Nsamples_per_graph must be greater than 1"):
        model.cate(intervention_idxs, intervention_values, Nsamples_per_graph=1)


@pytest.mark.parametrize(
    "variables",
    [
        five_cts_ungrouped_variables,
        six_cts_grouped_variables,
        cts_and_discrete_variables,
        mixed_type_group,
        two_cat_one_group,
    ],
)
def test_ite(tmpdir_factory, model_config, variables):

    model = DECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    base_samples = model.sample(Nsamples=10, most_likely_graph=False)

    intervention_idxs = torch.tensor([0])
    intervention_values = torch.tensor([0.0])

    ite0, _ = model.ite(base_samples, intervention_idxs, intervention_values, Ngraphs=100, most_likely_graph=False)

    ite1, _ = model.ite(base_samples, intervention_idxs, intervention_values, Ngraphs=100, most_likely_graph=False)

    #  Note that graph is deterministic here but samples are still random
    ite_det0, _ = model.ite(base_samples, intervention_idxs, intervention_values, Ngraphs=1, most_likely_graph=True)

    ite_det1, _ = model.ite(base_samples, intervention_idxs, intervention_values, Ngraphs=1, most_likely_graph=True)

    assert ite0.shape == base_samples.shape

    # Test values for continuous only
    assert np.any(ite0[variables.continuous_idxs] != 0)
    assert np.any(ite1[variables.continuous_idxs] != 0)
    assert np.any(ite_det0[variables.continuous_idxs] != 0)
    assert np.any(ite_det1[variables.continuous_idxs] != 0)
    # This test checks whether ITE's using different graphs are different.
    # This seems to fail?
    # assert np.all((ite0 - ite1)[variables.continuous_idxs] != 0)
    # This test checks whether ITE's using the same graph are the same.
    np.testing.assert_allclose(ite_det0, ite_det1)

    intervention_idxs = torch.tensor([0])
    intervention_values = torch.tensor([10.0])
    reference_values = torch.tensor([1.0])

    ite, _ = model.ite(
        base_samples,
        intervention_idxs,
        intervention_values,
        reference_values=reference_values,
        most_likely_graph=True,
        Ngraphs=1,
    )

    np.testing.assert_array_almost_equal(
        ite[:, intervention_idxs], (intervention_values - reference_values).expand(len(ite))
    )


@pytest.mark.parametrize(
    "variables",
    [
        five_cts_ungrouped_variables,
        six_cts_grouped_variables,
        cts_and_discrete_variables,
        mixed_type_group,
        two_cat_one_group,
    ],
)
def test_ite_no_intervention(tmpdir_factory, model_config, variables):

    model = DECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    samples = model.sample(100, most_likely_graph=True, samples_per_graph=100)
    ite, _ = model.ite(samples, most_likely_graph=True, Ngraphs=1)
    assert abs(ite.max()) < 1e-5

    # Should still work, as no intervention is applied
    ite2, _ = model.ite(samples, most_likely_graph=False)
    assert abs(ite2.max()) < 1e-5


@pytest.mark.parametrize(
    "base_offset,deltas",
    [
        (0.0, torch.tensor([0.0, 0.0, 0.0])),
        (1.0, torch.tensor([0.0, 0.0, 0.0])),
        (0.0, torch.tensor([0.1, 0.5, 1.0])),
        (1.0, torch.tensor([1.0, 2.0, 3.0])),
        (1.0, torch.tensor([-1.0, 2.0, -1.0])),
    ],
)
def test_categorical_base_posterior(base_offset, deltas):
    def get_cat_sample(gumbels, deltas):
        maxes = (gumbels + deltas).max(-1, keepdim=True)[0]
        return (gumbels + deltas >= maxes).float()

    n = 5000
    cat = CategoricalLikelihood(3, "cpu")
    cat.base_logits.data[0] = base_offset
    samples = cat.sample(n)
    cat_samples = get_cat_sample(samples, deltas)
    posteriors = cat.posterior(cat_samples, deltas)
    cat_resamples = get_cat_sample(posteriors, deltas)

    # This is 8 sigma of a standard Gumbel variable
    eight_sigma = 8 * math.pi / math.sqrt(6 * n)
    assert abs(samples.mean() - posteriors.mean()) < eight_sigma
    assert (cat_samples == cat_resamples).all()


@pytest.mark.parametrize(
    "base_offset,deltas",
    [
        (0.0, torch.tensor([0.0, 0.0, 0.0])),
        (1.0, torch.tensor([0.0, 0.0, 0.0])),
        (0.0, torch.tensor([0.1, 0.5, 1.0])),
        (1.0, torch.tensor([1.0, 2.0, 3.0])),
        (1.0, torch.tensor([-1.0, 2.0, -1.0])),
    ],
)
def test_binary_base_posterior(base_offset, deltas):
    def get_bin_sample(gumbels, deltas):
        return (gumbels + deltas > 0).float()

    n = 5000
    binary = BinaryLikelihood(3, "cpu")
    binary.base_logits.data[0] = base_offset
    samples = binary.sample(n)
    bin_samples = get_bin_sample(samples, deltas)
    posteriors = binary.posterior(bin_samples, deltas)
    bin_resamples = get_bin_sample(posteriors, deltas)

    # This is 8 sigma of a Logistic variable
    eight_sigma = 8 * math.pi / math.sqrt(3 * n)
    assert abs(samples.mean() - posteriors.mean()) < eight_sigma
    assert (bin_samples == bin_resamples).all()


@pytest.mark.parametrize("variables", [five_cts_ungrouped_variables])
@pytest.mark.parametrize(
    "conditioning_idxs",
    [
        torch.tensor([1, 2, 3]),  # Multiple variables
        torch.tensor([3]),  # Single variable
        torch.tensor([[1, 3]]),  # 2D array of multiple variables
        torch.tensor([[1, 3], [1, 3]]),  # Batch of 2 examples
        torch.tensor([[[1]]]),  # 3D array of a single variable
        torch.tensor(1),  # Scalar selection of a single variable
    ],
)
def test_cate(tmpdir_factory, model_config, variables, conditioning_idxs):

    model = DECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    intervention_idxs = torch.tensor([0])
    intervention_values = torch.tensor([0.0])
    conditioning_values = torch.zeros_like(conditioning_idxs)

    effect_idxs = torch.tensor([2])

    cate0, cate0_norm = model.cate(
        intervention_idxs,
        intervention_values,
        conditioning_idxs=conditioning_idxs,
        conditioning_values=conditioning_values,
        effect_idxs=effect_idxs,
        Nsamples_per_graph=100,
        Ngraphs=3,
        most_likely_graph=False,
    )

    cate1, cate1_norm = model.cate(
        intervention_idxs,
        intervention_values,
        conditioning_idxs=conditioning_idxs,
        conditioning_values=conditioning_values,
        effect_idxs=effect_idxs,
        Nsamples_per_graph=100,
        Ngraphs=3,
        most_likely_graph=False,
    )

    #  Note that graph is deterministic here but samples are still random
    cate_det0, cate_det0_norm = model.cate(
        intervention_idxs,
        intervention_values,
        conditioning_idxs=conditioning_idxs,
        conditioning_values=conditioning_values,
        effect_idxs=effect_idxs,
        Nsamples_per_graph=100,
        Ngraphs=1,
        most_likely_graph=True,
    )

    cate_det1, cate_det1_norm = model.cate(
        intervention_idxs,
        intervention_values,
        conditioning_idxs=conditioning_idxs,
        conditioning_values=conditioning_values,
        effect_idxs=effect_idxs,
        Nsamples_per_graph=100,
        Ngraphs=1,
        most_likely_graph=True,
    )

    # Test values for continuous only
    assert np.all((cate0 - cate1) != 0)
    assert np.all((cate0_norm - cate1_norm) != 0)

    assert np.all((cate_det0 - cate_det1) != 0)
    assert np.all((cate_det0_norm - cate_det1_norm) != 0)

    # Test consistency across batch of duplicated values
    np.testing.assert_allclose(cate0, cate0.item(0), atol=1e-4)


@pytest.mark.parametrize(
    "adj_mats, num_nodes, err, expected",
    [
        [np.array([[0, 1], [1, 0]]), 2, AssertionError, None],
        [np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]]), 2, AssertionError, None],
        [np.array([[0, 1], [0, 0]]), 2, None, (np.array([[0, 1], [0, 0]]), np.ones(1))],
        [
            np.array([[[0, 1], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]], [[0, 1], [0, 0]]]),
            2,
            None,
            (np.array([[[0, 0], [1, 0]], [[0, 1], [0, 0]]]), np.array([0.25, 0.75])),
        ],
    ],
)
def test_process_adjacency_matrix(adj_mats, num_nodes, err, expected):
    if err is not None:
        with pytest.raises(err):
            process_adjacency_mats(adj_mats, num_nodes)
    else:
        adj_mats_proc, adj_weights_proc = process_adjacency_mats(adj_mats, num_nodes)
        assert (adj_mats_proc == expected[0]).all()
        assert (adj_weights_proc == expected[1]).all()


# pylint: disable=redefined-outer-name
@pytest.mark.parametrize(
    "variables,constraint_idxs",
    [
        (five_cts_ungrouped_variables, [(0, 1)]),
        (six_cts_grouped_variables, [(0, 1)]),
        (cts_and_discrete_variables, [(0, 1)]),
        (cts_and_discrete_variables, [(0, 1), (1, 0)]),
        (cts_and_discrete_variables, [(4, 3)]),
        (cts_and_discrete_variables, [(1, 0), (2, 0), (3, 0), (4, 0)]),
    ],
)
def test_graph_negative_constraint(variables, model_config, tmpdir_factory, constraint_idxs):

    constraint_matrix = torch.full((variables.num_groups, variables.num_groups), np.nan)
    for (i, j) in constraint_idxs:
        constraint_matrix[i, j] = 0.0

    model = DECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    model.set_graph_constraint(constraint_matrix)

    prior_graph_samples = model.get_adj_matrix(do_round=False, samples=100)
    for (i, j) in constraint_idxs:
        error = abs(prior_graph_samples)[:, i, j].sum()
        assert error == 0.0


@pytest.mark.parametrize(
    "variables,constraint_idxs",
    [
        (five_cts_ungrouped_variables, [(0, 1)]),
        (six_cts_grouped_variables, [(0, 1)]),
        (cts_and_discrete_variables, [(0, 1)]),
        (cts_and_discrete_variables, [(0, 1), (2, 0)]),
        (cts_and_discrete_variables, [(4, 3)]),
        (cts_and_discrete_variables, [(1, 0), (2, 0), (3, 0), (4, 0)]),
    ],
)
def test_graph_positive_constraint(variables, model_config, tmpdir_factory, constraint_idxs):

    constraint_matrix = torch.full((variables.num_groups, variables.num_groups), np.nan)
    for (i, j) in constraint_idxs:
        constraint_matrix[i, j] = 1.0

    model = DECI.create(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )
    model.set_graph_constraint(constraint_matrix)

    prior_graph_samples = model.get_adj_matrix(do_round=False, samples=100)
    for (i, j) in constraint_idxs:
        error = abs(1.0 - prior_graph_samples)[:, i, j].sum()
        assert error == 0.0


def test_optimal_policy(tmpdir_factory, model_config):
    model = DECI.create(
        model_id="model_id",
        variables=cts_and_discrete_variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        model_config_dict=model_config,
        device="cpu",
    )

    def objective_fn(*args):
        return model.ite(*args)[0][0, ...]

    X = torch.rand(10, 15)
    X[..., 2] = 1
    X[..., 6] = 1
    X[..., 10] = 1
    intervention_idxs = np.array([4])

    # Test without budget
    assignments, values = model.posterior_expected_optimal_policy(X, intervention_idxs, objective_fn)
    assert (assignments.sum(1) <= 1).all()

    # Test with budget
    assignments_budget, values_budget = model.posterior_expected_optimal_policy(
        X, intervention_idxs, objective_fn, budget=torch.tensor([1])
    )
    assert assignments_budget.sum() <= 1
    assert (values_budget <= values).all()

    # Test with most likely graph
    assignments_one_graph, values_one_graph = model.posterior_expected_optimal_policy(
        X, intervention_idxs, objective_fn, num_posterior_samples=1, most_likely_graph=True
    )
    assert (assignments_one_graph.sum(1) <= 1).all()

    # Test with excessive budget
    assignments_large_budget, values_large_budget = model.posterior_expected_optimal_policy(
        X, intervention_idxs, objective_fn, budget=torch.tensor([1000]), num_posterior_samples=1, most_likely_graph=True
    )
    assert (assignments_large_budget.sum(1) <= 1).all()
    assert np.allclose(values_one_graph, values_large_budget, atol=1e-6)

    # Test non-binary variable
    with pytest.raises(AssertionError) as excinfo:
        model.posterior_expected_optimal_policy(X, np.array([0]), objective_fn)
    assert "binary" in str(excinfo.value)
