import numpy as np
import pytest

from causica.baselines.do_why import DoWhy
from causica.datasets.variables import Variable, Variables


@pytest.fixture
def four_node_dag():
    variables = Variables(
        [
            Variable("a", True, "continuous", 0, 1),
            Variable("b", True, "continuous", 0, 1),
            Variable("c", True, "continuous", 0, 1),
            Variable("d", True, "continuous", 0, 1),
        ]
    )
    d = len(variables)

    dag = np.zeros((d, d))
    dag[1, 0] = 1
    dag[3, 1] = 1
    dag[3, 2] = 1

    return variables, dag


@pytest.fixture
def mixed_type_dag():
    variables = Variables(
        [
            Variable("a", True, "continuous", 0, 1),
            Variable("b", True, "continuous", 0, 1),
            Variable("c", True, "binary", 0, 1),
            Variable("d", True, "categorical", 0, 2),
        ]
    )
    d = len(variables)

    dag = np.zeros((d, d))
    dag[1, 0] = 1
    dag[3, 1] = 1
    dag[3, 2] = 1
    dag[2, 1] = 1

    return variables, dag


@pytest.fixture
def dml_dag():
    variables = Variables(
        [
            Variable("a", True, "continuous", 0, 1),
            Variable("b", True, "continuous", 0, 1),
            Variable("c", True, "continuous", 0, 1),
        ]
    )
    d = len(variables)

    dag = np.zeros((d, d))
    dag[1, 0] = 1
    dag[2, 0] = 1
    dag[2, 1] = 1

    return variables, dag


@pytest.fixture
def dmlplus_dag():
    variables = Variables(
        [
            Variable("a", True, "continuous", 0, 1),
            Variable("b", True, "continuous", 0, 1),
            Variable("c", True, "continuous", 0, 1),
            Variable("d", True, "continuous", 0, 1),
        ]
    )
    d = len(variables)

    dag = np.zeros((d, d))
    dag[1, 0] = 1
    dag[2, 0] = 1
    dag[2, 1] = 1
    dag[3, 1] = 1

    return variables, dag


@pytest.fixture
def dml3_dag():
    variables = Variables(
        [
            Variable("a", True, "continuous", 0, 1),
            Variable("b", True, "continuous", 0, 1),
            Variable("c", True, "continuous", 0, 1),
            Variable("d", True, "continuous", 0, 1),
        ]
    )
    d = len(variables)

    dag = np.zeros((d, d))
    dag[1, 0] = 1
    dag[2, 0] = 1
    dag[2, 1] = 1

    return variables, dag


# pylint: disable=redefined-outer-name


@pytest.fixture
def four_node_linear_model(four_node_dag, tmpdir_factory):

    variables, dag = four_node_dag
    d = len(variables)

    Nsamples = 5
    weights = np.exp(np.random.randn(Nsamples))
    weights = weights / weights.sum()

    train_data = np.random.randn(10, d)
    test_data = np.random.randn(10, d)

    model = DoWhy(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        adj_matrix=dag,
        linear=True,
        polynomial_order=2,
        polynomial_bias=True,
        bootstrap_samples=100,
        train_data=train_data,
        adj_matrix_samples=[dag for i in range(Nsamples)],
        adj_matrix_sample_weights=weights,
    )

    intervention_idxs = np.array([3])
    intervention_values = np.array([5])

    conditioning_idxs = np.array([0])
    conditioning_values = np.array([5])

    effect_idxs = np.array([1])

    reference_values = train_data.mean(axis=0)[intervention_idxs]

    return (
        model,
        intervention_idxs,
        intervention_values,
        reference_values,
        conditioning_idxs,
        conditioning_values,
        effect_idxs,
        test_data,
    )


@pytest.fixture
def mixed_type_linear_model(mixed_type_dag, tmpdir_factory):

    variables, dag = mixed_type_dag
    d = len(variables)

    Nsamples = 5
    weights = np.exp(np.random.randn(Nsamples))
    weights = weights / weights.sum()

    train_data = np.random.randn(62, d)
    train_data[:, 2] = train_data[:, 2] > 0
    train_data[:, 3] = (train_data[:, 3] > -1) + (train_data[:, 3] > 1)
    test_data = np.random.randn(62, d)
    test_data[:, 2] = test_data[:, 2] > 0
    test_data[:, 3] = (test_data[:, 3] > -1) + (test_data[:, 3] > 1)

    model = DoWhy(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        adj_matrix=dag,
        linear=True,
        polynomial_order=2,
        polynomial_bias=True,
        bootstrap_samples=100,
        train_data=train_data,
        adj_matrix_samples=[dag for i in range(Nsamples)],
        adj_matrix_sample_weights=weights,
    )

    intervention_idxs = np.array([2])
    intervention_values = np.array([1])
    # Current implementation: default reference value is first value encountered
    reference_values = train_data[0, [2]]
    effect_idxs = [1]

    conditioning_idxs = np.array([0])
    conditioning_values = np.array([1.2])

    return (
        model,
        intervention_idxs,
        intervention_values,
        reference_values,
        conditioning_idxs,
        conditioning_values,
        effect_idxs,
        test_data,
    )


@pytest.fixture
def four_node_nonlinear_model(four_node_dag, tmpdir_factory):

    variables, dag = four_node_dag
    d = len(variables)

    Nsamples = 5
    weights = np.exp(np.random.randn(Nsamples))
    weights = weights / weights.sum()

    train_data = np.random.randn(10, d)
    test_data = np.random.randn(10, d)

    model = DoWhy(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        adj_matrix=dag,
        linear=False,
        polynomial_order=2,
        polynomial_bias=True,
        bootstrap_samples=100,
        train_data=train_data,
        adj_matrix_samples=[dag for i in range(Nsamples)],
        adj_matrix_sample_weights=weights,
    )

    intervention_idxs = np.array([3])
    intervention_values = np.array([5])
    conditioning_idxs = np.array([0])
    conditioning_values = np.array([5])

    effect_idxs = np.array([1])
    reference_values = train_data.mean(axis=0)[intervention_idxs]

    return (
        model,
        intervention_idxs,
        intervention_values,
        reference_values,
        conditioning_idxs,
        conditioning_values,
        effect_idxs,
        test_data,
    )


@pytest.fixture
def mixed_type_nonlinear_model(mixed_type_dag, tmpdir_factory):

    variables, dag = mixed_type_dag
    d = len(variables)

    Nsamples = 5
    weights = np.exp(np.random.randn(Nsamples))
    weights = weights / weights.sum()

    train_data = np.random.randn(62, d)
    train_data[:, 2] = train_data[:, 2] > 0
    train_data[:, 3] = (train_data[:, 3] > -1) + (train_data[:, 3] > 1)
    test_data = np.random.randn(62, d)
    test_data[:, 2] = test_data[:, 2] > 0
    test_data[:, 3] = (test_data[:, 3] > -1) + (test_data[:, 3] > 1)

    model = DoWhy(
        model_id="model_id",
        variables=variables,
        save_dir=tmpdir_factory.mktemp("save_dir"),
        adj_matrix=dag,
        linear=False,
        polynomial_order=2,
        polynomial_bias=True,
        bootstrap_samples=100,
        train_data=train_data,
        adj_matrix_samples=[dag for i in range(Nsamples)],
        adj_matrix_sample_weights=weights,
    )

    intervention_idxs = np.array([2])
    intervention_values = np.array([1])
    # Current implementation: default reference value is first value encountered
    reference_values = train_data[0, [2]]
    effect_idxs = [1]

    conditioning_idxs = np.array([0])
    conditioning_values = np.array([1.2])

    return (
        model,
        intervention_idxs,
        intervention_values,
        reference_values,
        conditioning_idxs,
        conditioning_values,
        effect_idxs,
        test_data,
    )


@pytest.mark.parametrize("linear_model", ["four_node_linear_model", "mixed_type_linear_model"])
def test_linear_ate(linear_model, request):

    linear_model = request.getfixturevalue(linear_model)
    model, intervention_idxs, intervention_values, reference_values, _, _, effect_idxs, _ = linear_model

    # mean treatment baseline consistency

    ml_ate_default_control, ml_norm_ate_default_control = model.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        reference_values=None,
        effect_idxs=effect_idxs,
        most_likely_graph=True,
    )

    ml_ate, ml_norm_ate = model.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        reference_values=reference_values,
        effect_idxs=effect_idxs,
        most_likely_graph=True,
    )

    assert np.allclose(ml_ate_default_control, ml_ate)
    assert np.allclose(ml_norm_ate_default_control, ml_norm_ate)

    # multi graph sample consistency

    marginalised_ate, marginalised_norm_ate = model.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        reference_values=reference_values,
        effect_idxs=effect_idxs,
        most_likely_graph=False,
    )

    assert np.allclose(marginalised_ate, ml_ate)
    assert np.allclose(marginalised_norm_ate, ml_norm_ate)


@pytest.mark.parametrize("nonlinear_model", ["four_node_nonlinear_model", "mixed_type_nonlinear_model"])
def test_non_linear_ate(nonlinear_model, request):

    nonlinear_model = request.getfixturevalue(nonlinear_model)
    model, intervention_idxs, intervention_values, reference_values, _, _, effect_idxs, _ = nonlinear_model
    seed = 2021

    # mean treatment baseline consistency

    ml_ate_default_control, ml_norm_ate_default_control = model.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        reference_values=None,
        effect_idxs=effect_idxs,
        most_likely_graph=True,
        fixed_seed=seed,
    )

    ml_ate, ml_norm_ate = model.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        reference_values=reference_values,
        effect_idxs=effect_idxs,
        most_likely_graph=True,
        fixed_seed=seed,
    )

    assert np.allclose(ml_ate_default_control, ml_ate)
    assert np.allclose(ml_norm_ate_default_control, ml_norm_ate)


def test_linear_cate(four_node_linear_model):

    (
        model,
        intervention_idxs,
        intervention_values,
        reference_values,
        conditioning_idxs,
        conditioning_values,
        _,
        _,
    ) = four_node_linear_model

    # mean treatment baseline consistency

    ml_ate_default_control, ml_norm_ate_default_control = model.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=conditioning_idxs,
        conditioning_values=conditioning_values,
        reference_values=None,
        most_likely_graph=True,
    )

    ml_ate, ml_norm_ate = model.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=conditioning_idxs,
        conditioning_values=conditioning_values,
        reference_values=reference_values,
        most_likely_graph=True,
    )

    assert np.allclose(ml_ate_default_control, ml_ate)
    assert np.allclose(ml_norm_ate_default_control, ml_norm_ate)

    # multi graph sample consistency

    marginalised_ate, marginalised_norm_ate = model.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=conditioning_idxs,
        conditioning_values=conditioning_values,
        reference_values=reference_values,
        most_likely_graph=False,
    )

    assert np.allclose(marginalised_ate, ml_ate)
    assert np.allclose(marginalised_norm_ate, ml_norm_ate)


def test_non_linear_cate(four_node_nonlinear_model):

    (
        model,
        intervention_idxs,
        intervention_values,
        reference_values,
        conditioning_idxs,
        conditioning_values,
        effect_idxs,
        _,
    ) = four_node_nonlinear_model
    seed = 2021

    # mean treatment baseline consistency

    ml_ate_default_control, ml_norm_ate_default_control = model.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=conditioning_idxs,
        conditioning_values=conditioning_values,
        effect_idxs=effect_idxs,
        reference_values=None,
        most_likely_graph=True,
        fixed_seed=seed,
    )

    ml_ate, ml_norm_ate = model.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=conditioning_idxs,
        conditioning_values=conditioning_values,
        effect_idxs=effect_idxs,
        reference_values=reference_values,
        most_likely_graph=True,
        fixed_seed=seed,
    )

    assert np.allclose(ml_ate_default_control, ml_ate)
    assert np.allclose(ml_norm_ate_default_control, ml_norm_ate)


@pytest.mark.parametrize("linear,most_likely_graph", [(True, True), (True, False), (False, True), (False, False)])
def test_ate_causal_consistency(dml_dag, dmlplus_dag, tmpdir_factory, linear, most_likely_graph):

    seed = 2021
    Nsamples = 5
    weights = np.exp(np.random.randn(Nsamples))
    weights = weights / weights.sum()

    # Generate data for the larger graph
    train_data = np.random.randn(10, 4)

    intervention_idxs = np.array([1])
    intervention_values = np.array([3])
    reference_values = train_data.mean(axis=0)[intervention_idxs]

    model_dml = DoWhy(
        model_id="model_id",
        variables=dml_dag[0],
        save_dir=tmpdir_factory.mktemp("save_dir"),
        adj_matrix=dml_dag[1],
        linear=linear,
        polynomial_order=2,
        polynomial_bias=True,
        bootstrap_samples=100,
        train_data=train_data[:, :3],
        adj_matrix_samples=[dml_dag[1] for _ in range(Nsamples)],
        adj_matrix_sample_weights=weights,
    )

    model_dmlplus = DoWhy(
        model_id="model_id",
        variables=dmlplus_dag[0],
        save_dir=tmpdir_factory.mktemp("save_dir"),
        adj_matrix=dmlplus_dag[1],
        linear=linear,
        polynomial_order=2,
        polynomial_bias=True,
        bootstrap_samples=100,
        train_data=train_data,
        adj_matrix_samples=[dmlplus_dag[1] for _ in range(Nsamples)],
        adj_matrix_sample_weights=weights,
    )

    ml_ate_dml, ml_norm_ate_dml = model_dml.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        reference_values=None,
        most_likely_graph=most_likely_graph,
        fixed_seed=seed,
    )

    ml_ate_dmlplus, ml_norm_ate_dmlplus = model_dmlplus.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        reference_values=reference_values,
        most_likely_graph=most_likely_graph,
        fixed_seed=seed,
    )

    assert np.allclose(ml_ate_dml, ml_ate_dmlplus[:3])
    assert np.allclose(ml_norm_ate_dml, ml_norm_ate_dmlplus[:3])


@pytest.mark.parametrize("linear,most_likely_graph", [(True, True), (True, False), (False, True), (False, False)])
def test_ate_causal_consistency2(dml3_dag, dmlplus_dag, tmpdir_factory, linear, most_likely_graph):

    seed = 2021
    Nsamples = 5
    weights = np.exp(np.random.randn(Nsamples))
    weights = weights / weights.sum()

    # Generate data for the larger graph
    train_data = np.random.randn(10, 4)

    intervention_idxs = np.array([1])
    intervention_values = np.array([3])

    model_dml = DoWhy(
        model_id="model_id",
        variables=dml3_dag[0],
        save_dir=tmpdir_factory.mktemp("save_dir"),
        adj_matrix=dml3_dag[1],
        linear=linear,
        polynomial_order=2,
        polynomial_bias=True,
        bootstrap_samples=100,
        train_data=train_data,
        adj_matrix_samples=[dml3_dag[1] for _ in range(Nsamples)],
        adj_matrix_sample_weights=weights,
        parallel_n_jobs=1,
    )

    model_dmlplus = DoWhy(
        model_id="model_id",
        variables=dmlplus_dag[0],
        save_dir=tmpdir_factory.mktemp("save_dir"),
        adj_matrix=dmlplus_dag[1],
        linear=linear,
        polynomial_order=2,
        polynomial_bias=True,
        bootstrap_samples=100,
        train_data=train_data,
        adj_matrix_samples=[dmlplus_dag[1] for _ in range(Nsamples)],
        adj_matrix_sample_weights=weights,
        parallel_n_jobs=1,
    )

    # NOTE: Only testing the effects on everything but variable 4 because DoWhy miscalculates the effect for ml_ate_dmlplus.
    ml_ate_dml, ml_norm_ate_dml = model_dml.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        reference_values=None,
        most_likely_graph=most_likely_graph,
        fixed_seed=seed,
    )

    ml_ate_dmlplus, ml_norm_ate_dmlplus = model_dmlplus.cate(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        reference_values=None,
        most_likely_graph=most_likely_graph,
        fixed_seed=seed,
    )

    assert np.allclose(ml_ate_dml, ml_ate_dmlplus)
    assert np.allclose(ml_norm_ate_dml, ml_norm_ate_dmlplus)


def test_linear_log_prob(four_node_linear_model):
    # instantiate model

    model, intervention_idxs, intervention_values, _, _, _, _, test_data = four_node_linear_model
    seed = 2021

    ml_log_prob = model.log_prob(
        X=test_data,
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        most_likely_graph=True,
        fixed_seed=seed,
    )

    # multi graph sample consistency
    marginalised_log_prob = model.log_prob(
        X=test_data,
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        most_likely_graph=False,
        fixed_seed=seed,
    )

    assert np.allclose(ml_log_prob, marginalised_log_prob)


@pytest.mark.parametrize("linear,most_likely_graph", [(True, True), (True, False), (False, True), (False, False)])
def test_log_prob_causal_consistency(dml3_dag, dmlplus_dag, tmpdir_factory, linear, most_likely_graph):

    seed = 2021
    Nsamples = 5 if linear else 2
    weights = np.exp(np.random.randn(Nsamples))
    weights = weights / weights.sum()

    # Generate data for the larger graph
    train_data = np.random.randn(10, 4)
    test_data = np.random.randn(10, 4)

    intervention_idxs = np.array([1])
    intervention_values = np.array([3])

    model_dml3 = DoWhy(
        model_id="model_id",
        variables=dml3_dag[0],
        save_dir=tmpdir_factory.mktemp("save_dir"),
        adj_matrix=dml3_dag[1],
        linear=linear,
        polynomial_order=2,
        polynomial_bias=True,
        bootstrap_samples=100,
        train_data=train_data,
        adj_matrix_samples=[dml3_dag[1] for _ in range(Nsamples)],
        adj_matrix_sample_weights=weights,
        parallel_n_jobs=1,
    )

    model_dmlplus = DoWhy(
        model_id="model_id",
        variables=dmlplus_dag[0],
        save_dir=tmpdir_factory.mktemp("save_dir"),
        adj_matrix=dmlplus_dag[1],
        linear=linear,
        polynomial_order=2,
        polynomial_bias=True,
        bootstrap_samples=100,
        train_data=train_data,
        adj_matrix_samples=[dmlplus_dag[1] for _ in range(Nsamples)],
        adj_matrix_sample_weights=weights,
        parallel_n_jobs=1,
    )

    log_prob_dml3 = model_dml3.log_prob(
        X=test_data,
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        most_likely_graph=most_likely_graph,
        fixed_seed=seed,
    )

    log_prob_dmlplus = model_dmlplus.log_prob(
        X=test_data,
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values,
        conditioning_idxs=None,
        conditioning_values=None,
        most_likely_graph=most_likely_graph,
        fixed_seed=seed,
    )

    assert np.allclose(log_prob_dml3, log_prob_dmlplus)
