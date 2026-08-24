import numpy as np
import pandas as pd
import pytest

from linearmodels import (
    BetweenOLS as UpBetweenOLS,
    FirstDifferenceOLS as UpFirstDifferenceOLS,
    PanelOLS as UpPanelOLS,
    PooledOLS as UpPooledOLS,
    RandomEffects as UpRandomEffects,
)
from mojo_linearmodels import (
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)
from mojo_linearmodels._lib import (
    demean,
    first_difference,
    group_mean,
    native_predict,
    quasi_demean,
)


@pytest.fixture(scope="module")
def balanced():
    rng = np.random.default_rng(812)
    entities, periods = 36, 9
    index = pd.MultiIndex.from_product(
        [range(entities), range(periods)], names=["firm", "year"]
    )
    n = len(index)
    x1, x2 = rng.normal(size=(2, n))
    entity_effect = np.repeat(rng.normal(size=entities), periods)
    time_effect = np.tile(rng.normal(scale=0.5, size=periods), entities)
    y = (
        0.8
        + 1.7 * x1
        - 0.45 * x2
        + entity_effect
        + time_effect
        + rng.normal(scale=0.35, size=n)
    )
    exog = pd.DataFrame({"const": 1.0, "x1": x1, "x2": x2}, index=index)
    no_constant = exog[["x1", "x2"]]
    weights = pd.Series(rng.uniform(0.2, 2.0, n), index=index)
    return pd.Series(y, index=index, name="y"), exog, no_constant, weights


@pytest.fixture(scope="module")
def unbalanced(balanced):
    y, x, no_constant, weights = balanced
    rng = np.random.default_rng(91)
    keep = rng.random(len(y)) > 0.18
    return y[keep], x[keep], no_constant[keep], weights[keep]


def assert_result_parity(ours, upstream, *, covariance=True, rtol=2e-10):
    assert np.allclose(ours.params, upstream.params, rtol=rtol, atol=2e-11)
    if covariance:
        assert np.allclose(
            ours.std_errors,
            upstream.std_errors,
            rtol=rtol,
            atol=2e-11,
            equal_nan=True,
        )
    assert ours.rsquared == pytest.approx(upstream.rsquared, rel=2e-10, abs=2e-11)
    assert ours.nobs == upstream.nobs


@pytest.mark.parametrize("cov_type", ["unadjusted", "robust"])
def test_pooled_ols(balanced, cov_type):
    y, x, _, _ = balanced
    assert_result_parity(
        PooledOLS(y, x).fit(cov_type=cov_type),
        UpPooledOLS(y, x).fit(cov_type=cov_type),
    )


def test_pooled_ols_weights(balanced):
    y, x, _, weights = balanced
    assert_result_parity(
        PooledOLS(y, x, weights=weights).fit(cov_type="robust"),
        UpPooledOLS(y, x, weights=weights).fit(cov_type="robust"),
    )


@pytest.mark.parametrize(
    "effects",
    [
        {"entity_effects": True},
        {"time_effects": True},
        {"entity_effects": True, "time_effects": True},
    ],
)
def test_panel_effects(balanced, effects):
    y, x, _, _ = balanced
    assert_result_parity(
        PanelOLS(y, x, **effects).fit(cov_type="robust"),
        UpPanelOLS(y, x, **effects).fit(cov_type="robust"),
    )


def test_panel_two_way_unbalanced(unbalanced):
    y, x, _, _ = unbalanced
    effects = {"entity_effects": True, "time_effects": True}
    assert_result_parity(
        PanelOLS(y, x, **effects).fit(),
        UpPanelOLS(y, x, **effects).fit(),
        rtol=2e-9,
    )


def test_panel_weighted_unbalanced(unbalanced):
    y, x, _, weights = unbalanced
    effects = {"entity_effects": True, "time_effects": True}
    assert_result_parity(
        PanelOLS(y, x, weights=weights, **effects).fit(cov_type="robust"),
        UpPanelOLS(y, x, weights=weights, **effects).fit(cov_type="robust"),
        rtol=2e-9,
    )


def test_panel_clustered_entity(balanced):
    y, x, _, _ = balanced
    ours = PanelOLS(y, x, entity_effects=True).fit(
        cov_type="clustered", cluster_entity=True
    )
    upstream = UpPanelOLS(y, x, entity_effects=True).fit(
        cov_type="clustered", cluster_entity=True
    )
    assert_result_parity(ours, upstream)


def test_panel_two_way_clustered(balanced):
    y, x, _, _ = balanced
    ours = PanelOLS(
        y, x, entity_effects=True, time_effects=True
    ).fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    upstream = UpPanelOLS(
        y, x, entity_effects=True, time_effects=True
    ).fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    assert_result_parity(ours, upstream)


def test_between_ols(balanced):
    y, x, _, _ = balanced
    assert_result_parity(BetweenOLS(y, x).fit(), UpBetweenOLS(y, x).fit())


def test_between_weighted_reweight(unbalanced):
    y, x, _, weights = unbalanced
    assert_result_parity(
        BetweenOLS(y, x, weights=weights).fit(reweight=True, cov_type="robust"),
        UpBetweenOLS(y, x, weights=weights).fit(
            reweight=True, cov_type="robust"
        ),
    )


def test_first_difference(balanced):
    y, _, x, _ = balanced
    assert_result_parity(
        FirstDifferenceOLS(y, x).fit(cov_type="robust"),
        UpFirstDifferenceOLS(y, x).fit(cov_type="robust"),
    )


def test_first_difference_weighted_unbalanced(unbalanced):
    y, _, x, weights = unbalanced
    assert_result_parity(
        FirstDifferenceOLS(y, x, weights=weights).fit(),
        UpFirstDifferenceOLS(y, x, weights=weights).fit(),
    )


def test_first_difference_rejects_constant(balanced):
    y, x, _, _ = balanced
    with pytest.raises(ValueError):
        FirstDifferenceOLS(y, x)


@pytest.mark.parametrize("rows", [31, 65_537])
def test_first_difference_simd_tail_and_parallel_threshold(rows):
    columns = 7
    values = np.arange((rows + 1) * columns, dtype=np.float64).reshape(
        rows + 1, columns
    )
    right = np.arange(1, rows + 1, dtype=np.int64)
    actual = first_difference(values, right)
    assert np.array_equal(actual, values[1:] - values[:-1])


@pytest.mark.parametrize("rows", [31, 65_537])
def test_predict_simd_tail_and_parallel_threshold(rows):
    rng = np.random.default_rng(rows)
    values = rng.normal(size=(rows, 7))
    beta = rng.normal(size=7)
    assert np.allclose(native_predict(values, beta), values @ beta, rtol=2e-15)


@pytest.mark.parametrize("rows", [31, 65_537])
def test_quasi_demean_simd_tail_and_parallel_threshold(rows):
    rng = np.random.default_rng(rows + 1)
    columns, groups = 7, 13
    values = rng.normal(size=(rows, columns))
    weights = rng.uniform(0.2, 2.0, size=rows)
    codes = np.arange(rows, dtype=np.int64) % groups
    means = rng.normal(size=(groups, columns))
    theta = rng.uniform(size=groups)
    expected = (
        values * np.sqrt(weights)[:, None]
        - theta[codes, None] * means[codes]
    )
    assert np.allclose(
        quasi_demean(values, weights, codes, means, theta), expected,
        rtol=2e-15,
    )


@pytest.mark.parametrize("sorted_codes", [True, False])
def test_group_kernels_parallel_sorted_and_unsorted_fallback(sorted_codes):
    rng = np.random.default_rng(97 + sorted_codes)
    rows, columns, groups = 40_001, 7, 17
    values = rng.normal(size=(rows, columns))
    weights = rng.uniform(0.2, 2.0, size=rows)
    if sorted_codes:
        codes = np.arange(rows, dtype=np.int64) * groups // rows
    else:
        codes = np.arange(rows, dtype=np.int64) % groups
    sums = np.bincount(codes, weights=weights, minlength=groups)
    expected_means = np.zeros((groups, columns))
    np.add.at(expected_means, codes, values * weights[:, None])
    expected_means /= sums[:, None]
    means, actual_sums = group_mean(values, weights, codes, groups)
    assert np.allclose(actual_sums, sums, rtol=2e-15)
    assert np.allclose(means, expected_means, rtol=3e-15)
    expected_demeaned = (
        values - expected_means[codes]
    ) * np.sqrt(weights)[:, None]
    assert np.allclose(
        demean(values, weights, codes, groups), expected_demeaned,
        rtol=3e-15,
    )


def test_random_effects(balanced):
    y, x, _, _ = balanced
    assert_result_parity(
        RandomEffects(y, x).fit(cov_type="robust"),
        UpRandomEffects(y, x).fit(cov_type="robust"),
    )


def test_random_effects_weighted_unbalanced(unbalanced):
    y, x, _, weights = unbalanced
    assert_result_parity(
        RandomEffects(y, x, weights=weights).fit(),
        UpRandomEffects(y, x, weights=weights).fit(),
    )


def test_panel_formula_and_result_prediction(balanced):
    y, x, _, _ = balanced
    data = x.assign(y=y)
    ours = PanelOLS.from_formula(
        "y ~ 1 + x1 + x2 + EntityEffects", data
    ).fit()
    upstream = UpPanelOLS.from_formula(
        "y ~ 1 + x1 + x2 + EntityEffects", data
    ).fit()
    assert_result_parity(ours, upstream)
    assert np.allclose(
        ours.predict().fitted_values,
        upstream.predict().fitted_values,
        atol=2e-11,
    )
    assert np.allclose(
        ours.model.predict(ours.params, data=data).to_numpy(),
        upstream.model.predict(upstream.params, data=data).to_numpy(),
        atol=2e-11,
    )


def test_missing_rows_are_dropped(balanced):
    y, x, _, _ = balanced
    y = y.copy()
    x = x.copy()
    y.iloc[3] = np.nan
    x.iloc[17, 1] = np.nan
    with pytest.warns(Warning):
        upstream = UpPooledOLS(y, x).fit()
    ours = PooledOLS(y, x).fit()
    assert_result_parity(ours, upstream)


def test_unsupported_adjustments_raise(balanced):
    y, x, _, _ = balanced
    with pytest.raises(NotImplementedError, match="small-sample"):
        RandomEffects(y, x).fit(small_sample=True)
    with pytest.raises(NotImplementedError, match="group_debias"):
        PooledOLS(y, x).fit(
            cov_type="clustered",
            cluster_entity=True,
            cluster_time=True,
            group_debias=True,
        )


def test_singleton_effects_raise():
    index = pd.MultiIndex.from_tuples(
        [("a", 0), ("a", 1), ("b", 0)], names=["firm", "year"]
    )
    y = pd.Series([1.0, 2.0, 3.0], index=index)
    x = pd.DataFrame({"const": 1.0, "x": [0.0, 1.0, 2.0]}, index=index)
    with pytest.raises(NotImplementedError, match="singleton"):
        PanelOLS(y, x, entity_effects=True).fit()
