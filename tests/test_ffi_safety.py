import numpy as np
import pytest

from mojo_linearmodels._lib import (
    demean,
    first_difference,
    group_mean,
    native_iv,
    native_ols,
    native_predict,
)


def test_native_ols_validates_shape_length_and_values():
    x = np.ones((4, 2))
    with pytest.raises(ValueError, match="length 4"):
        native_ols(x, np.ones(3))
    with pytest.raises(ValueError, match="two-dimensional"):
        native_ols(np.ones(4), np.ones(4))
    with pytest.raises(ValueError, match="finite"):
        native_ols(np.array([[1.0], [np.nan]]), np.ones(2))


def test_native_predict_validates_coefficient_length():
    with pytest.raises(ValueError, match="length 2"):
        native_predict(np.ones((3, 2)), np.ones(3))


def test_group_kernels_validate_codes_and_lengths():
    x = np.ones((3, 2))
    weights = np.ones(3)
    with pytest.raises(ValueError, match=r"\[0, groups\)"):
        group_mean(x, weights, np.array([0, 1, 2]), 2)
    with pytest.raises(ValueError, match="length 3"):
        demean(x, np.ones(2), np.array([0, 0, 0]), 1)
    with pytest.raises(ValueError, match=r"\[0, groups\)"):
        demean(x, weights, np.array([0, -1, 0]), 1)


def test_first_difference_validates_source_rows_and_empty_result():
    x = np.arange(12.0).reshape(4, 3)
    with pytest.raises(ValueError, match="between 1"):
        first_difference(x, np.array([0]))
    with pytest.raises(ValueError, match="between 1"):
        first_difference(x, np.array([4]))
    assert first_difference(x, np.array([], dtype=np.int64)).shape == (0, 3)


def test_native_iv_validates_matching_rows():
    with pytest.raises(ValueError, match="same number of rows"):
        native_iv(np.ones((4, 1)), np.ones(4), np.ones((3, 1)))


def test_float_conversion_rejects_precision_loss():
    with pytest.raises(ValueError, match="exact float64"):
        native_ols(
            np.array([[2**53 + 1], [1]], dtype=np.int64),
            np.array([1, 2]),
        )
