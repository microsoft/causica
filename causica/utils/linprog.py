from typing import Optional

import numpy as np
import scipy.optimize


def col_row_constrained_lin_prog(
    objective_matrix: np.ndarray, colwise_budget: Optional[np.ndarray] = None, **scipy_kwargs
):
    """Solves the following linear program using `scipy`

    max Σᵢⱼ AᵢⱼOᵢⱼ
    subject to
        Σᵢ Aᵢⱼ ≤ bⱼ for each column j
        Σⱼ Aᵢⱼ ≤ 1 for each row i
        0 ≤ Aᵢⱼ ≤ 1

    where `O` is the objective matrix, `b` is the budget and `A` is the assignment matrix.

    Args:
        objective_matrix: an array of objective values
        colwise_budget: if passed, this should be an array of shape equal to the number of columns
            of the objective. If `None`, no columnwise constraints are applied.
        scipy_kwargs: additional arguments to be consumed by `scipy.optimize.linprog`.
    Returns:
        assignments: the optimal assignments, correctly reshaped to match the shape of
            `objective_matrix`
        res: the optimal value attained
    """
    # Set up a linear program in scipy.optimize.linprog formulation
    n_row, n_col = objective_matrix.shape
    dtype = objective_matrix.dtype
    cost_vector = -objective_matrix.flatten()

    row_constraint_matrix = np.kron(np.eye(n_row), np.ones(n_col))
    row_constraint_value = np.ones(n_row, dtype=dtype)
    if colwise_budget is None:
        constraint_matrix = row_constraint_matrix
        constraint_value = row_constraint_value
    else:
        col_constraint_matrix = np.kron(np.ones(n_row), np.eye(n_col))

        constraint_matrix = np.concatenate([row_constraint_matrix, col_constraint_matrix], axis=0)
        constraint_value = np.concatenate([row_constraint_value, colwise_budget], axis=0)

    res = scipy.optimize.linprog(
        cost_vector, A_ub=constraint_matrix, b_ub=constraint_value, bounds=(0, 1), **scipy_kwargs
    )
    assignments = res.x.reshape(objective_matrix.shape)
    return assignments, -res.fun
