import warnings

import numpy as np

from causica.datasets.variables import Variable
from causica.preprocessing.transforms import IdentityTransform, UnitScaler


def test_identity_transform():
    transform = IdentityTransform()
    data = np.random.rand(100, 6)
    transformed = transform.fit_transform(data)
    np.testing.assert_allclose(data, transformed)
    restored = transform.inverse_transform(transformed)
    np.testing.assert_allclose(data, restored)


def test_unit_scaler():
    transform = UnitScaler([Variable("numeric_input", True, "continuous", 0, 9)])
    data = np.arange(10, dtype=np.float32).reshape((10, 1))
    transformed = transform.fit_transform(data)
    np.testing.assert_almost_equal(transformed.min(), 0.0)
    np.testing.assert_almost_equal(transformed.max(), 1.0)
    restored = transform.inverse_transform(transformed)
    np.testing.assert_allclose(data, restored)

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        transform = UnitScaler(
            [Variable("same", True, "continuous", 1234, 1234), Variable("diff", True, "continuous", 1234, 4321)]
        )
        assert len(w) == 1
        assert (
            str(w[-1].message)
            == "Variable with name 'same' has the same upper and lower values. Is this variable a constant?"
        )
