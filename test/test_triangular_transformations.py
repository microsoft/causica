import pytest
import torch

from causica.triangular_transformations import fill_triangular, unfill_triangular

DIM = 5


@pytest.mark.parametrize("batch_size", [tuple(), (3,), (4, 2)])
def test_fill_unfill(batch_size):
    """Test that filling and unfilling results in the same tensor"""
    matrix = torch.randn(batch_size + (DIM, DIM))
    lower_vec = unfill_triangular(matrix, upper=False)
    upper_vec = unfill_triangular(matrix, upper=True)

    torch.testing.assert_close(torch.tril(matrix, diagonal=-1), fill_triangular(lower_vec, upper=False))
    torch.testing.assert_close(torch.triu(matrix, diagonal=1), fill_triangular(upper_vec, upper=True))
    torch.testing.assert_close(
        torch.tril(matrix, diagonal=-1).transpose(-2, -1), fill_triangular(lower_vec, upper=True)
    )
    torch.testing.assert_close(
        torch.triu(matrix, diagonal=1).transpose(-2, -1), fill_triangular(upper_vec, upper=False)
    )


@pytest.mark.parametrize("batch_size", [tuple(), (3,), (4, 2)])
def test_unfill_fill(batch_size):
    """Test that unfilling and filling results in the same tensor"""
    vec = torch.randn(batch_size + (DIM * (DIM - 1) // 2,))
    lower_mat = fill_triangular(vec, upper=False)
    upper_mat = fill_triangular(vec, upper=True)

    torch.testing.assert_close(unfill_triangular(lower_mat, upper=False), vec)
    torch.testing.assert_close(unfill_triangular(upper_mat, upper=True), vec)
    torch.testing.assert_close(unfill_triangular(lower_mat, upper=True), torch.zeros_like(vec))
    torch.testing.assert_close(unfill_triangular(upper_mat, upper=False), torch.zeros_like(vec))
