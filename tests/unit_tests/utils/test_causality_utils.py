from math import sqrt

import numpy as np

from causica.datasets.intervention_data import InterventionData
from causica.models.imodel import IModelForCounterfactuals
from causica.utils import causality_utils
from causica.datasets.variables import Variable, Variables
from causica.utils.nri_utils import (
    convert_temporal_to_static_adjacency_matrix,
    make_temporal_adj_matrix_compatible,
)


def test_approximate_maximal_acyclic_subgraph():
    non_dag = np.zeros((3, 3))
    non_dag[0, 1] = 1
    non_dag[1, 2] = 1
    non_dag[2, 0] = 1
    sub_dag = causality_utils.approximate_maximal_acyclic_subgraph(non_dag)

    assert causality_utils.dag_pen_np(non_dag) != 0.0
    assert (causality_utils.dag_pen_np(sub_dag)) == 0
    assert sub_dag.sum() >= 0.5 * non_dag.sum()


def test_cpdag2dags_high_order_cycles():
    """
    The only non-colider solutions to this CPDAG have order 2 cycles so only determined edges should be returned
    """
    cp_dag = np.ones((4, 4))
    cp_dag[np.diag_indices(4)] = 0
    cp_dag[0, 3] = 0
    cp_dag[3, 0] = 0
    cp_dag[1, 2] = 0
    cp_dag[2, 1] = 0

    proposed_solutions = causality_utils.cpdag2dags(cp_dag)
    assert proposed_solutions.sum() == 0


def test_cpdag2dags_already_dag():
    cp_dag = np.zeros((3, 3))
    cp_dag[0, 1] = 1
    cp_dag[1, 2] = 1

    proposed_solutions = causality_utils.cpdag2dags(cp_dag)

    # 0 <- 1 -> 2
    solution0 = cp_dag.copy()

    assert proposed_solutions.shape[0] == 1
    assert len(proposed_solutions.shape) == 3

    for dag in proposed_solutions:

        assert causality_utils.dag_pen_np(dag) == 0.0
        assert np.all(dag == solution0)


def test_cpdag2dags_simple_chain():

    # 0 - 1 - 2
    cp_dag = np.zeros((3, 3))
    cp_dag[0, 1] = 1
    cp_dag[1, 0] = 1
    cp_dag[1, 2] = 1
    cp_dag[2, 1] = 1

    proposed_solutions = causality_utils.cpdag2dags(cp_dag)

    # 0 <- 1 -> 2
    solution0 = cp_dag.copy()
    solution0[2, 1] = 0
    solution0[0, 1] = 0

    # 0 -> 1 -> 2
    solution1 = cp_dag.copy()
    solution1[2, 1] = 0
    solution1[1, 0] = 0

    # 0 <- 1 <- 2
    solution2 = cp_dag.copy()
    solution2[1, 2] = 0
    solution2[0, 1] = 0

    assert proposed_solutions.shape[0] == 3

    for dag in proposed_solutions:

        assert causality_utils.dag_pen_np(dag) == 0.0
        assert np.all(dag == solution0) or np.all(dag == solution1) or np.all(dag == solution2)


def test_cpdag2dags_keep_extra_collider():

    # this test ensures that the method works in the presence of an extra collider

    # 0 - 1 - 2 -> 3 <- 4
    cp_dag = np.zeros((5, 5))
    cp_dag[0, 1] = 1
    cp_dag[1, 0] = 1
    cp_dag[1, 2] = 1
    cp_dag[2, 1] = 1

    cp_dag[2, 3] = 1
    cp_dag[4, 3] = 1

    proposed_solutions = causality_utils.cpdag2dags(cp_dag)

    # 0 <- 1 -> 2 -> 3 <- 4
    solution0 = cp_dag.copy()
    solution0[2, 1] = 0
    solution0[0, 1] = 0

    # 0 -> 1 -> 2 -> 3 <- 4
    solution1 = cp_dag.copy()
    solution1[2, 1] = 0
    solution1[1, 0] = 0

    # 0 <- 1 <- 2 -> 3 <- 4
    solution2 = cp_dag.copy()
    solution2[1, 2] = 0
    solution2[0, 1] = 0

    assert proposed_solutions.shape[0] == 3

    for dag in proposed_solutions:

        assert causality_utils.dag_pen_np(dag) == 0.0
        assert np.all(dag == solution0) or np.all(dag == solution1) or np.all(dag == solution2)


def test_cpdag2dags_sampling():

    # this test ensures that the method works in the presence of an extra collider

    # 0 - 1 - 2 -> 3 <- 4
    cp_dag = np.zeros((5, 5))
    cp_dag[0, 1] = 1
    cp_dag[1, 0] = 1
    cp_dag[1, 2] = 1
    cp_dag[2, 1] = 1

    cp_dag[2, 3] = 1
    cp_dag[4, 3] = 1

    # 0 <- 1 -> 2 -> 3 <- 4
    solution0 = cp_dag.copy()
    solution0[2, 1] = 0
    solution0[0, 1] = 0

    # 0 -> 1 -> 2 -> 3 <- 4
    solution1 = cp_dag.copy()
    solution1[2, 1] = 0
    solution1[1, 0] = 0

    # 0 <- 1 <- 2 -> 3 <- 4
    solution2 = cp_dag.copy()
    solution2[1, 2] = 0
    solution2[0, 1] = 0

    proposed_solutions = causality_utils.cpdag2dags(cp_dag, samples=1)
    assert proposed_solutions.shape[0] == 1

    for dag in proposed_solutions:

        assert causality_utils.dag_pen_np(dag) == 0.0
        assert np.all(dag == solution0) or np.all(dag == solution1) or np.all(dag == solution2)

    proposed_solutions = causality_utils.cpdag2dags(cp_dag, samples=2)
    assert proposed_solutions.shape[0] == 2

    for dag in proposed_solutions:

        assert causality_utils.dag_pen_np(dag) == 0.0
        assert np.all(dag == solution0) or np.all(dag == solution1) or np.all(dag == solution2)

    proposed_solutions = causality_utils.cpdag2dags(cp_dag, samples=200)
    assert proposed_solutions.shape[0] == 3

    for dag in proposed_solutions:

        assert causality_utils.dag_pen_np(dag) == 0.0
        assert np.all(dag == solution0) or np.all(dag == solution1) or np.all(dag == solution2)


def test_cpdag2dags_dont_add_extra_collider():

    # this test ensures that the method does not add an extra collider

    # 0 - 1 - 2 <- 3
    cp_dag = np.zeros((4, 4))
    cp_dag[0, 1] = 1
    cp_dag[1, 0] = 1
    cp_dag[1, 2] = 1
    cp_dag[2, 1] = 1

    cp_dag[3, 2] = 1

    proposed_solutions = causality_utils.cpdag2dags(cp_dag)

    # 0 <- 1 <- 2 <- 3
    solution0 = cp_dag.copy()
    solution0[1, 2] = 0
    solution0[0, 1] = 0

    assert proposed_solutions.shape[0] == 1

    for dag in proposed_solutions:

        assert causality_utils.dag_pen_np(dag) == 0.0
        assert np.all(dag == solution0)


def test_normalise_data():
    arr_1 = np.array([[10, 6], [11, 10], [121, 55]])
    arr_2 = np.array([[7, 4], [5, 15], [1, 72]])
    var_1 = Variable("1", True, "continuous", 0, 200, group_name="Group 1")
    var_2 = Variable("2", True, "categorical", 0, 100, group_name="Group 2")
    variables = Variables([var_1, var_2])
    arr_1_norm, arr_2_norm = causality_utils.normalise_data(arrs=[arr_1, arr_2], variables=variables, processed=False)
    np.testing.assert_array_equal(np.array([[0.05, 6], [0.055, 10], [0.605, 55]]), arr_1_norm)
    np.testing.assert_array_equal(np.array([[0.035, 4], [0.025, 15], [0.005, 72]]), arr_2_norm)

    arr_1 = np.array([[10], [11], [12]])
    var_1 = Variable("1", True, "continuous", 5, 20, group_name="Group 1")
    variables = Variables([var_1])
    [arr_1_norm] = causality_utils.normalise_data(arrs=[arr_1], variables=variables, processed=False)

    np.testing.assert_array_equal(np.array([[5 / 15], [0.4], [7 / 15]]), arr_1_norm)


def test_get_ite_from_samples():
    intervention_samples = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    reference_samples = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]])

    ite = causality_utils.get_ite_from_samples(intervention_samples, reference_samples)
    expected = np.array([[-10, -10, -10], [-10, -10, -10], [-10, -10, -10]])
    np.testing.assert_equal(ite, expected)

    ite = causality_utils.get_ite_from_samples(reference_samples, intervention_samples)
    expected = np.array([[10, 10, 10], [10, 10, 10], [10, 10, 10]])
    np.testing.assert_equal(ite, expected)

    intervention_samples = np.array([[0, 3, 5, 6], [0, 6, 8, 1]])

    reference_samples = np.array([[4, 8, 9, 2], [1, 2, 3, 3]])

    variables = Variables(
        [
            Variable("1", False, "continuous", 1, 10),
            Variable("2", False, "continuous", 2, 8),
            Variable("3", False, "continuous", 2, 11),
            Variable("4", False, "continuous", 1, 7),
        ]
    )

    result = causality_utils.get_ite_from_samples(
        intervention_samples, reference_samples, variables, normalise=False, processed=True
    )
    np.testing.assert_array_equal(result, np.array([[-4, -5, -4, 4], [-1, 4, 5, -2]]))

    result = causality_utils.get_ite_from_samples(
        intervention_samples, reference_samples, variables, normalise=True, processed=True
    )

    for i in range(intervention_samples.shape[0]):
        for j in range(intervention_samples.shape[1]):
            assert result[i, j] == (intervention_samples[i, j] - variables[j].lower) / (
                variables[j].upper - variables[j].lower
            ) - (reference_samples[i, j] - variables[j].lower) / (variables[j].upper - variables[j].lower)


def test_calculate_rmse():
    a = np.array([2, -2, 5])
    b = np.array([3, 2, 7])
    assert causality_utils.calculate_rmse(a, b) == sqrt(7)


def test_calculate_per_group_rmse():

    a = np.array([[0, 4, 1], [2, 2, 4]])
    b = np.array([[0, 3, 1], [2, 2, 2]])

    variable_1 = Variable("1", True, "continuous", 0, 10, group_name="Group 1")
    variable_2 = Variable("2", True, "continuous", 0, 10, group_name="Group 1")
    variable_3 = Variable("3", True, "continuous", 0, 1, group_name="Group 2")

    variables = Variables([variable_1, variable_2, variable_3])

    per_group_rmse = causality_utils.calculate_per_group_rmse(a, b, variables)

    np.testing.assert_equal(per_group_rmse, np.array([[sqrt(0.5), 0], [0, 2]]))


def test_filter_effect_columns():

    processed_data = np.array([[9, 0, 0, 1]])

    variable_1 = Variable("1", True, "continuous", 0, 10, group_name="Group 1")
    variable_2 = Variable("2", True, "categorical", 0, 2, group_name="Group 1")
    variables = Variables([variable_1, variable_2])

    effect_idxs = np.array([1])

    [filtered_data], filtered_variables = causality_utils.filter_effect_columns(
        [processed_data], variables, effect_idxs, processed=True
    )

    np.testing.assert_equal(filtered_data, np.array([[0, 0, 1]]))
    assert filtered_variables[0].name == "2"

    unprocessed_data_1 = np.array([[9, 7, 8]])
    unprocessed_data_2 = np.array([[9, 7, 8], [4, 6, 1]])
    variable_1 = Variable("1", True, "continuous", 0, 10, group_name="Group 1")
    variable_2 = Variable("2", True, "continuous", 0, 2, group_name="Group 2")
    variable_3 = Variable("3", True, "continuous", 0, 2, group_name="Group 3")
    variables = Variables([variable_1, variable_2, variable_3])

    effect_idxs = np.array([1, 2])

    [filtered_data_1, filtered_data_2], filtered_variables = causality_utils.filter_effect_columns(
        [unprocessed_data_1, unprocessed_data_2], variables, effect_idxs, processed=False
    )

    np.testing.assert_equal(filtered_data_1, np.array([[7, 8]]))
    np.testing.assert_equal(filtered_data_2, np.array([[7, 8], [6, 1]]))
    assert filtered_variables[0].name == "2"
    assert filtered_variables[1].name == "3"


def test_get_ite_evaluation_results():
    class MockDeci(IModelForCounterfactuals):
        def __init__(self, ret_val):
            super().__init__(model_id="MockDeci", variables=Variables([]), save_dir="")
            self.ret_val = ret_val

        def ite(self, *_args, **_kwargs):
            return self.ret_val

    MockDeci.__abstractmethods__ = set()
    # pylint: disable=abstract-class-instantiated
    mock_deci = MockDeci((np.array([[0, 0, 0.5], [0, 0, 0.5]]), np.array([[0, 0, 0.5], [0, 0, 0.5]])))

    conditioning_idxs = np.array([0, 1, 2])
    intervention_idxs = np.array([0])
    intervention_values_1 = np.array([1])
    intervention_reference_1 = np.array([-1])

    conditioning_values = [[0, 4, 6], [6, 2, 0]]

    intervened_data = np.array([[1, 1, 2], [1, 2, 3]])

    reference_data = np.array([[1, 1, 3], [1, 2, 4]])

    intervention_data = InterventionData(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values_1,
        test_data=intervened_data,
        conditioning_idxs=conditioning_idxs,
        conditioning_values=conditioning_values,
        intervention_reference=intervention_reference_1,
        reference_data=reference_data,
    )

    counterfactual_datasets = [intervention_data]

    variable_1 = Variable("1", True, "continuous", -1, 10, group_name="Group 1")
    variable_2 = Variable("2", True, "continuous", 0, 10, group_name="Group 2")
    variable_3 = Variable("3", True, "continuous", 0, 1, group_name="Group 3")

    variables = Variables([variable_1, variable_2, variable_3])

    ite_eval_results, _ = causality_utils.get_ite_evaluation_results(
        mock_deci, counterfactual_datasets, variables, processed=True, most_likely_graph=False
    )

    assert ite_eval_results.all == 0.5
    assert ite_eval_results.across_groups == np.array([0.5])
    np.testing.assert_equal(ite_eval_results.across_interventions, np.array([0, 0, 1.5]))

    counterfactual_datasets = [intervention_data, intervention_data]
    ite_eval_results, _ = causality_utils.get_ite_evaluation_results(
        mock_deci, counterfactual_datasets, variables, processed=True, most_likely_graph=False
    )
    assert ite_eval_results.all == 0.5
    np.testing.assert_equal(ite_eval_results.across_groups, np.array([0.5, 0.5]))

    intervention_data = InterventionData(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values_1,
        test_data=intervened_data,
        conditioning_idxs=conditioning_idxs,
        conditioning_values=conditioning_values,
        intervention_reference=intervention_reference_1,
        reference_data=reference_data,
        effect_idxs=np.array([1]),
    )

    counterfactual_datasets = [intervention_data]

    ite_eval_results, _ = causality_utils.get_ite_evaluation_results(
        mock_deci, counterfactual_datasets, variables, processed=True, most_likely_graph=False
    )

    assert ite_eval_results.all == 0.0
    assert ite_eval_results.n_groups == 1
    np.testing.assert_equal(ite_eval_results.across_groups, np.array([0.0]))

    mock_deci = MockDeci(
        (
            np.array([[0, 0, 0.5, 0], [0, 0, 0.5, 0]]),
            np.array([[0, 0, 0.5, 0], [0, 0, 0.5, 0]]),
        )
    )

    conditioning_idxs = np.array([0, 1])
    intervention_idxs = np.array([0])
    intervention_values_1 = np.array([1])
    intervention_reference_1 = np.array([-1])

    conditioning_values = np.array([[0, 0, 0, 1], [6, 0, 0, 1]])

    intervened_data = np.array([[1, 0, 0, 1], [1, 0, 0, 1]])

    reference_data = np.array([[1, 0, 0, 1], [1, 0, 0, 1]])

    intervention_data = InterventionData(
        intervention_idxs=intervention_idxs,
        intervention_values=intervention_values_1,
        test_data=intervened_data,
        conditioning_idxs=conditioning_idxs,
        conditioning_values=conditioning_values,
        intervention_reference=intervention_reference_1,
        reference_data=reference_data,
    )

    counterfactual_datasets = [intervention_data]

    variable_1 = Variable("1", True, "continuous", -1, 10, group_name="Group 1")
    variable_2 = Variable("2", True, "categorical", 0, 2, group_name="Group 2")

    variables = Variables([variable_1, variable_2])

    ite_eval_results, _ = causality_utils.get_ite_evaluation_results(
        mock_deci, counterfactual_datasets, variables, processed=True, most_likely_graph=False
    )

    assert ite_eval_results.all == 0
    assert ite_eval_results.n_interventions == 1
    assert ite_eval_results.n_groups == 2
    assert ite_eval_results.n_samples == 2


def test_make_temporal_adj_matrix_compatible():
    # is_static=False, both temporal, lag1>lag2, no batch
    adj_1 = np.array([[[1, 0], [0, 1]], [[0, 1], [0, 1]], [[0, 1], [1, 0]]])
    adj_2 = np.array([[[0, 1], [0, 1]], [[1, 1], [1, 0]]])
    adj_1_comp, adj_2_comp = make_temporal_adj_matrix_compatible(
        adj_1, adj_2, is_static=False, adj_matrix_2_lag=adj_2.shape[0] - 1
    )
    assert len(adj_1_comp.shape) == 3
    assert len(adj_2_comp.shape) == 3
    assert adj_1_comp.shape[0] == 3 and adj_2_comp.shape[0] == 3
    assert adj_1_comp.shape[1] == adj_2_comp.shape[1]
    assert np.array_equal(adj_1_comp[1, ...], np.array([[0, 1], [0, 1]]))

    # is_static=False, both temporal, lag1<lag2, no batch
    adj_1_comp, adj_2_comp = make_temporal_adj_matrix_compatible(
        adj_2, adj_1, is_static=False, adj_matrix_2_lag=adj_1.shape[0] - 1
    )
    assert len(adj_1_comp.shape) == 3
    assert len(adj_2_comp.shape) == 3
    assert adj_1_comp.shape[0] == 3 and adj_2_comp.shape[0] == 3
    assert adj_1_comp.shape[1] == adj_2_comp.shape[1]
    assert np.array_equal(adj_1_comp[1, ...], np.array([[1, 1], [1, 0]]))

    # is_static=False, lag1>lag2, batch
    adj_1 = np.array(
        [[[[1, 0], [0, 1]], [[0, 1], [0, 1]], [[0, 1], [1, 0]]], [[[1, 0], [0, 1]], [[0, 1], [0, 1]], [[0, 1], [1, 0]]]]
    )  # [2,3,2,2]
    adj_2 = np.array(
        [
            [[1, 1, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0]],
            [[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]],
        ]
    )  # [2,4,4]
    adj_1_comp, adj_2_comp = make_temporal_adj_matrix_compatible(adj_1, adj_2, is_static=True, adj_matrix_2_lag=1)
    assert len(adj_1_comp.shape) == 4 and len(adj_2_comp.shape) == 3
    assert adj_1_comp.shape[0] == 2 and adj_2_comp.shape[0] == 2
    assert adj_1_comp.shape[1] == 3 and adj_2_comp.shape[1] == 6
    assert adj_1_comp.shape[2] == 2 and adj_2_comp.shape[2] == 6
    assert np.array_equal(adj_1_comp[0, 1, ...], np.array([[0, 1], [0, 1]]))
    assert np.array_equal(
        adj_2_comp[1, ...],
        np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 1],
            ]
        ),
    )

    # is_static=False, lag1<lag2, batch
    adj_1 = np.array([[[[0, 1], [0, 1]], [[1, 1], [1, 0]]], [[[0, 1], [0, 1]], [[1, 1], [1, 0]]]])
    adj_2 = np.array(
        [
            [
                [1, 1, 0, 0, 1, 1],
                [1, 1, 1, 0, 0, 1],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1],
            ],
            [
                [1, 0, 0, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
        ]
    )
    adj_1_comp, adj_2_comp = make_temporal_adj_matrix_compatible(adj_1, adj_2, is_static=True, adj_matrix_2_lag=2)
    assert len(adj_1_comp.shape) == 4 and len(adj_2_comp.shape) == 3
    assert adj_1_comp.shape[0] == 2 and adj_2_comp.shape[0] == 2
    assert adj_1_comp.shape[1] == 3 and adj_2_comp.shape[1] == 6
    assert adj_1_comp.shape[2] == 2 and adj_2_comp.shape[2] == 6
    assert np.array_equal(adj_1_comp[0, 1, ...], np.array([[1, 1], [1, 0]]))
    assert np.array_equal(
        adj_2_comp[1, ...],
        np.array(
            [
                [1, 0, 0, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        ),
    )


def test_convert_temporal_adj_matrix_to_static():
    # auto_regressive conversion
    # Setup the true adj_matrix
    true_adj = np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]], [[1, 1], [0, 1]]])  # [3,2,2]
    # Setup pred adj
    adj_mat_2 = np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]]])  # [2,2,2]

    true_adj, adj_mat_2 = make_temporal_adj_matrix_compatible(true_adj, adj_mat_2, is_static=False)

    static_true = convert_temporal_to_static_adjacency_matrix(true_adj, conversion_type="auto_regressive", fill_value=9)
    static_pred = convert_temporal_to_static_adjacency_matrix(
        adj_mat_2, conversion_type="auto_regressive", fill_value=9
    )
    target_true = np.array(
        [
            [9, 9, 9, 9, 1, 1],
            [9, 9, 9, 9, 0, 1],
            [9, 9, 9, 9, 1, 1],
            [9, 9, 9, 9, 0, 0],
            [9, 9, 9, 9, 0, 1],
            [9, 9, 9, 9, 1, 0],
        ]
    )
    target_pred = np.array(
        [
            [9, 9, 9, 9, 0, 0],
            [9, 9, 9, 9, 0, 0],
            [9, 9, 9, 9, 1, 1],
            [9, 9, 9, 9, 0, 0],
            [9, 9, 9, 9, 0, 1],
            [9, 9, 9, 9, 1, 0],
        ]
    )
    assert np.array_equal(static_true, target_true)
    assert np.array_equal(static_pred, target_pred)
    # Varlingam model with full_time conversion
    # Setup the true adj_matrix
    true_adj = np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]], [[1, 1], [0, 1]]])  # [3,2,2]
    # Setup pred adj
    adj_mat_2 = np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]]])  # [2,2,2]

    true_adj, adj_mat_2 = make_temporal_adj_matrix_compatible(true_adj, adj_mat_2, is_static=False)

    static_true = convert_temporal_to_static_adjacency_matrix(true_adj, conversion_type="full_time", fill_value=9)
    static_pred = convert_temporal_to_static_adjacency_matrix(adj_mat_2, conversion_type="full_time", fill_value=9)
    target_true = np.array(
        [
            [0, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [9, 9, 0, 1, 1, 1],
            [9, 9, 1, 0, 0, 0],
            [9, 9, 9, 9, 0, 1],
            [9, 9, 9, 9, 1, 0],
        ]
    )
    target_pred = np.array(
        [
            [0, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [9, 9, 0, 1, 1, 1],
            [9, 9, 1, 0, 0, 0],
            [9, 9, 9, 9, 0, 1],
            [9, 9, 9, 9, 1, 0],
        ]
    )
    assert np.array_equal(static_true, target_true)
    assert np.array_equal(static_pred, target_pred)
    # fold-time model, adj_true lag is smaller with conversion_type = 'full_time'
    adj_mat_2 = np.zeros((2, 8, 8))  # fold-time format:[N, (lag2+1)*node, (lag2+1)*node]

    true_adj, adj_mat_2 = make_temporal_adj_matrix_compatible(true_adj, adj_mat_2, is_static=True, adj_matrix_2_lag=3)

    static_true = convert_temporal_to_static_adjacency_matrix(true_adj, conversion_type="full_time")

    target_true = np.array(
        [
            [0, 1, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
    )
    assert np.array_equal(static_true, target_true)

    # fold-time model, adj_true lag is larger
    true_adj = np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]], [[1, 1], [0, 1]]])  # [3,2,2]
    adj_mat_2 = np.array(
        [
            [[0, 1, 1, 1], [1, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0]],
            [[1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 1]],
        ]
    )
    adj_mat_2_target = np.array(
        [
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 1, 1],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 1],
            ],
        ]
    )

    _, adj_mat_2 = make_temporal_adj_matrix_compatible(true_adj, adj_mat_2, is_static=True, adj_matrix_2_lag=1)

    assert np.array_equal(adj_mat_2, adj_mat_2_target)
