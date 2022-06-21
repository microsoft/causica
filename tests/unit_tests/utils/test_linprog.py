import numpy as np
import pytest

from causica.utils.linprog import col_row_constrained_lin_prog


@pytest.mark.parametrize(
    "objective_matrix,budget,expected_assignment,expected_value",
    [
        (np.array([[0, 1], [0, 0], [2, 0]]), np.array([1, 1]), np.array([[0, 1], [0, 0], [1, 0]]), 3),
        (np.array([[0, 1], [0, 0], [2, 0]]), None, None, 3),
        (
            np.array([[0, 1, 1], [0, 0, 1], [2, 0, 1]]),
            np.array([1, 1, 1]),
            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
            4,
        ),
        (
            np.array([[0, 1], [0, 0], [2, 0], [3, 1], [3, 3]]),
            np.array([2, 2]),
            np.array([[0, 1], [0, 0], [1, 0], [1, 0], [0, 1]]),
            9,
        ),
        (np.array([[0, 1], [0, 0], [2, 0]]), np.array([0, 0]), np.array([[0, 0], [0, 0], [0, 0]]), 0),
        (np.array([[0, 1], [0, 0], [2, 0]]), np.array([10, 10]), None, 3),
    ],
)
def test_col_row_constrained_lin_prog(objective_matrix, budget, expected_assignment, expected_value):
    assignment, value = col_row_constrained_lin_prog(objective_matrix, budget)
    if expected_assignment is not None:  # Sometimes there are multiple optimal assignments
        assert np.allclose(assignment, expected_assignment)
    assert np.allclose(value, expected_value)
