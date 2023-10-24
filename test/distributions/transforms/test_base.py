import gc

from causica.distributions.transforms.base import TypedTransform


class _StrIntTransform(TypedTransform[str, int]):
    """Dummy transform for testing types."""

    def _call(self, x: str) -> int:
        return int(x)

    def _inverse(self, y: int) -> str:
        return str(y)


def test_typed_transform() -> None:
    """Tests the basic functionality of TypedTransform.

    Should be paired with a mypy step to ensure that the types are consistent."""

    transform = _StrIntTransform()
    x: str = "1"
    y: int = transform(x)
    inverse: TypedTransform[int, str] = transform.inv  # Check that the inverse is indeed recognized as a TypedTransform
    assert isinstance(inverse, TypedTransform)
    assert transform.inv(y) == x


def test_weak_ref_inv_release() -> None:
    """Test that the weak reference to the inverse is released."""
    transform = _StrIntTransform()
    inverse = transform.inv
    id_ = id(inverse)

    # Check that the inverse is kept while there is an active reference
    assert id(transform.inv) == id_

    # Release the direct reference to the inverse and force garbage collection
    del inverse
    gc.collect()

    # Check that accessing the inverse now yields a new object
    assert id(transform.inv) != id_
