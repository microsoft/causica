import pytest
import torch
from tensordict import TensorDict, TensorDictBase

from causica.datasets.tensordict_utils import expand_tensordict_groups, tensordict_shapes, unbind_values, unbound_items


def _assert_tensordict_allclose(a: TensorDictBase, b: TensorDictBase) -> None:
    paired_items = zip(
        a.items(include_nested=True, leaves_only=True),
        b.items(include_nested=True, leaves_only=True),
    )
    assert all(
        key_a == key_b and torch.allclose(value_a, value_b) for (key_a, value_a), (key_b, value_b) in paired_items
    )


def test_expand_tensordict_groups():
    # Create a TensorDict with a group of variables
    td = TensorDict({"group": torch.ones((2, 3)), "categorical": torch.ones((2, 4))}, batch_size=2)

    # Define the variable groups
    variable_groups = {"group": ["var1", "var2", "var3"], "categorical": ["categorical"]}

    # Expand the TensorDict
    expanded_td = expand_tensordict_groups(td, variable_groups)

    # Check that the expanded TensorDict has the correct variables
    assert set(expanded_td.keys()) == {"var1", "var2", "var3", "categorical"}

    # Check that the expanded variables have the correct shape
    assert expanded_td["var1"].shape == (2, 1)
    assert expanded_td["var2"].shape == (2, 1)
    assert expanded_td["var3"].shape == (2, 1)
    assert expanded_td["categorical"].shape == (2, 4)


def test_unbound_items():
    td = TensorDict({"a": torch.tensor([[[3]], [[4]]])}, batch_size=2)
    assert dict(unbound_items(td, dim=0)) == {
        ("a", "0"): torch.tensor([[3]]),
        ("a", "1"): torch.tensor([[4]]),
    }


def test_unbind_values():
    td = TensorDict({"a": torch.rand(2, 3, 4), "b": torch.rand(2, 3, 5)}, batch_size=2)

    unbound_positive = unbind_values(td, dim=1)
    unbound_negative = unbind_values(td, dim=-2)
    assert unbound_positive.batch_size == unbound_negative.batch_size == torch.Size([2])

    expected_shapes = {
        ("a", "0"): torch.Size([4]),
        ("a", "1"): torch.Size([4]),
        ("a", "2"): torch.Size([4]),
        ("b", "0"): torch.Size([5]),
        ("b", "1"): torch.Size([5]),
        ("b", "2"): torch.Size([5]),
    }
    assert tensordict_shapes(unbound_positive) == expected_shapes
    _assert_tensordict_allclose(unbound_positive, unbound_negative)


def test_unbind_values_along_batch_dim():
    td = TensorDict({"a": torch.rand(2, 3, 4), "b": torch.rand(2, 3, 5)}, batch_size=2)
    expected_shapes = {
        ("a", "0"): torch.Size([3, 4]),
        ("a", "1"): torch.Size([3, 4]),
        ("b", "0"): torch.Size([3, 5]),
        ("b", "1"): torch.Size([3, 5]),
    }
    unbound_positive = unbind_values(td, dim=0)
    unbound_negative = unbind_values(td, dim=-3)
    assert unbound_positive.batch_size == unbound_negative.batch_size == torch.Size([])
    assert tensordict_shapes(unbound_positive) == expected_shapes
    _assert_tensordict_allclose(unbound_positive, unbound_negative)


def test_unbind_values_different_dims():
    td = TensorDict({"a": torch.rand(1, 2), "b": torch.rand(1, 2, 3), "c": torch.rand(1, 2, 3, 4)}, batch_size=1)
    unbound = unbind_values(td, dim=0)
    assert unbound.batch_size == torch.Size([])
    assert tensordict_shapes(unbound) == {
        ("a", "0"): torch.Size([2]),
        ("b", "0"): torch.Size([2, 3]),
        ("c", "0"): torch.Size([2, 3, 4]),
    }

    unbound = unbind_values(td, dim=1)
    assert unbound.batch_size == torch.Size([1])
    assert tensordict_shapes(unbound) == {
        ("a", "0"): torch.Size([]),
        ("a", "1"): torch.Size([]),
        ("b", "0"): torch.Size([3]),
        ("b", "1"): torch.Size([3]),
        ("c", "0"): torch.Size([3, 4]),
        ("c", "1"): torch.Size([3, 4]),
    }

    unbound = unbind_values(td, dim=-1)
    assert unbound.batch_size == torch.Size([1])
    assert tensordict_shapes(unbound) == {
        ("a", "0"): torch.Size([]),
        ("a", "1"): torch.Size([]),
        ("b", "0"): torch.Size([2]),
        ("b", "1"): torch.Size([2]),
        ("b", "2"): torch.Size([2]),
        ("c", "0"): torch.Size([2, 3]),
        ("c", "1"): torch.Size([2, 3]),
        ("c", "2"): torch.Size([2, 3]),
        ("c", "3"): torch.Size([2, 3]),
    }

    # We cannot unbind along an axis that will be a batch dimension for at least one samples, when it won't be the same
    # batch dimension for all samples.
    with pytest.raises(IndexError):
        unbound = unbind_values(td, dim=-2)
    with pytest.raises(IndexError):
        unbound = unbind_values(td, dim=-4)
    with pytest.raises(IndexError):
        unbound = unbind_values(td, dim=2)
