import numpy as np
import torch

from causica.graph.dag_constraint import calculate_dagness


def test_calculate_dagness():

    dag = torch.Tensor([[0, 0, 1], [0, 0, 1], [0, 0, 0]])

    assert calculate_dagness(dag) == 0

    dag_one_cycle = torch.Tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]])

    dag_two_cycle = torch.Tensor([[1, 1, 0], [0, 0, 1], [1, 0, 0]])

    assert calculate_dagness(dag_one_cycle) > 0
    assert calculate_dagness(dag_two_cycle) > 0
    assert calculate_dagness(dag_one_cycle) < calculate_dagness(dag_two_cycle)

    self_loop_non_dag = torch.Tensor([[1, 0], [0, 1]])
    np.testing.assert_almost_equal(
        calculate_dagness(self_loop_non_dag),
        2 * np.exp(1) - 2,
        decimal=5,
    )
