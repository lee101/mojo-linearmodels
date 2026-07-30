import numpy as np
import pandas as pd
import pytest

from linearmodels.iv import IV2SLS as UpIV2SLS
from linearmodels.iv import IVLIML as UpIVLIML
from mojo_linearmodels.iv import IV2SLS, IVLIML


@pytest.fixture(scope="module")
def iv_data():
    rng = np.random.default_rng(124)
    n = 1800
    z1, z2, exog, confounder = rng.normal(size=(4, n))
    endog = 0.9 * z1 + 0.45 * z2 + 0.8 * confounder + rng.normal(size=n)
    y = 1.2 + 0.55 * exog + 2.1 * endog + confounder + rng.normal(size=n)
    index = pd.Index([f"obs-{i}" for i in range(n)])
    dependent = pd.Series(y, index=index, name="y")
    exogenous = pd.DataFrame({"const": 1.0, "x": exog}, index=index)
    endogenous = pd.DataFrame({"endog": endog}, index=index)
    instruments = pd.DataFrame({"z1": z1, "z2": z2}, index=index)
    weights = pd.Series(rng.uniform(0.3, 1.8, n), index=index)
    return dependent, exogenous, endogenous, instruments, weights


def assert_iv_parity(ours, upstream, *, rtol=3e-10):
    assert np.allclose(ours.params, upstream.params, rtol=rtol, atol=2e-11)
    assert np.allclose(
        ours.std_errors, upstream.std_errors, rtol=rtol, atol=2e-11
    )
    assert ours.rsquared == pytest.approx(upstream.rsquared, rel=rtol)
    assert ours.kappa == pytest.approx(upstream.kappa, rel=rtol, abs=2e-12)


@pytest.mark.parametrize("cov_type", ["robust", "unadjusted"])
def test_iv2sls(iv_data, cov_type):
    y, x, endog, instruments, _ = iv_data
    assert_iv_parity(
        IV2SLS(y, x, endog, instruments).fit(cov_type=cov_type),
        UpIV2SLS(y, x, endog, instruments).fit(cov_type=cov_type),
    )


def test_iv2sls_debiased(iv_data):
    y, x, endog, instruments, _ = iv_data
    assert_iv_parity(
        IV2SLS(y, x, endog, instruments).fit(debiased=True),
        UpIV2SLS(y, x, endog, instruments).fit(debiased=True),
    )


def test_iv2sls_weights(iv_data):
    y, x, endog, instruments, weights = iv_data
    assert_iv_parity(
        IV2SLS(y, x, endog, instruments, weights=weights).fit(),
        UpIV2SLS(y, x, endog, instruments, weights=weights).fit(),
    )


def test_ivliml(iv_data):
    y, x, endog, instruments, _ = iv_data
    assert_iv_parity(
        IVLIML(y, x, endog, instruments).fit(),
        UpIVLIML(y, x, endog, instruments).fit(),
        rtol=2e-9,
    )


def test_ivliml_fuller(iv_data):
    y, x, endog, instruments, _ = iv_data
    assert_iv_parity(
        IVLIML(y, x, endog, instruments, fuller=1).fit(cov_type="unadjusted"),
        UpIVLIML(y, x, endog, instruments, fuller=1).fit(
            cov_type="unadjusted"
        ),
        rtol=2e-9,
    )


def test_ivliml_user_kappa(iv_data):
    y, x, endog, instruments, _ = iv_data
    assert_iv_parity(
        IVLIML(y, x, endog, instruments, kappa=1.01).fit(),
        UpIVLIML(y, x, endog, instruments, kappa=1.01).fit(),
    )


def test_iv_two_way_clustered(iv_data):
    y, x, endog, instruments, _ = iv_data
    clusters = np.column_stack(
        [np.arange(len(y)) % 30, np.arange(len(y)) % 17]
    )
    assert_iv_parity(
        IV2SLS(y, x, endog, instruments).fit(
            cov_type="clustered", clusters=clusters
        ),
        UpIV2SLS(y, x, endog, instruments).fit(
            cov_type="clustered", clusters=clusters
        ),
    )


def test_iv_formula(iv_data):
    y, x, endog, instruments, _ = iv_data
    data = pd.concat([y, x[["x"]], endog, instruments], axis=1)
    formula = "y ~ 1 + x + [endog ~ z1 + z2]"
    ours = IV2SLS.from_formula(formula, data).fit()
    upstream = UpIV2SLS.from_formula(formula, data).fit()
    assert_iv_parity(ours, upstream)
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


def test_just_identified_iv():
    rng = np.random.default_rng(77)
    n = 600
    z = rng.normal(size=n)
    v = rng.normal(size=n)
    endog = z + v
    y = 2.0 * endog + v + rng.normal(size=n)
    ours = IV2SLS(y, None, endog, z).fit()
    upstream = UpIV2SLS(y, None, endog, z).fit()
    assert_iv_parity(ours, upstream)
