from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .._data import frame
from .._lib import native_iv, native_predict
from .._results import RegressionResults


def _iv_data(dependent, exog, endog, instruments, weights):
    y = frame(dependent, "dependent")
    index = y.index
    e = frame(exog, "exog").reindex(index) if exog is not None else pd.DataFrame(index=index)
    d = frame(endog, "endog").reindex(index) if endog is not None else pd.DataFrame(index=index)
    q = (
        frame(instruments, "instruments").reindex(index)
        if instruments is not None
        else pd.DataFrame(index=index)
    )
    if weights is None:
        w = pd.Series(1.0, index=index, name="__weight__")
    else:
        w = frame(weights, "weights").iloc[:, 0].reindex(index).rename("__weight__")
    all_data = pd.concat([y, e, d, q, w], axis=1).dropna()
    if (all_data["__weight__"] <= 0).any():
        raise ValueError("weights must be strictly positive")
    ne, nd, nq = e.shape[1], d.shape[1], q.shape[1]
    pos = 1
    yv = all_data.iloc[:, 0].to_numpy(dtype=float)
    ev = all_data.iloc[:, pos:pos + ne].to_numpy(dtype=float)
    pos += ne
    dv = all_data.iloc[:, pos:pos + nd].to_numpy(dtype=float)
    pos += nd
    qv = all_data.iloc[:, pos:pos + nq].to_numpy(dtype=float)
    wv = all_data["__weight__"].to_numpy(dtype=float, copy=True)
    wv /= wv.mean()
    x = np.ascontiguousarray(np.column_stack([ev, dv]))
    z = np.ascontiguousarray(np.column_stack([ev, qv]))
    names = list(e.columns.astype(str)) + list(d.columns.astype(str))
    return (
        np.ascontiguousarray(yv),
        x,
        z,
        np.ascontiguousarray(wv),
        all_data.index,
        names,
        ne,
    )


def _cluster_sum(scores: np.ndarray, groups) -> np.ndarray:
    codes, unique = pd.factorize(groups, sort=False)
    sums = np.zeros((len(unique), scores.shape[1]))
    np.add.at(sums, codes, scores)
    return sums.T @ sums


def _iv_cov(x, y, z, beta, kappa, cov_type, debiased, config):
    n, k = x.shape
    pinvz_x = np.linalg.lstsq(z, x, rcond=None)[0]
    projected_x = z @ pinvz_x
    v = (1 - kappa) * (x.T @ x / n) + kappa * (x.T @ projected_x / n)
    bread = np.linalg.inv(v)
    eps = y - x @ beta
    scale = n / (n - k) if debiased else 1.0
    key = cov_type.lower().replace("-", "")
    if key in {"unadjusted", "homoskedastic"}:
        score_cov = scale * (eps @ eps / n) * v
    elif key in {"robust", "heteroskedastic"}:
        scores = projected_x * eps[:, None]
        score_cov = scale * scores.T @ scores / n
    elif key in {"clustered", "cluster"}:
        clusters = config.get("clusters")
        if clusters is None:
            raise ValueError("clustered covariance requires clusters")
        clusters = np.asarray(clusters)
        scores = projected_x * eps[:, None]
        if clusters.ndim == 1:
            meat = _cluster_sum(scores, clusters)
        elif clusters.ndim == 2 and clusters.shape[1] == 2:
            joined = pd.MultiIndex.from_arrays(
                [clusters[:, 0], clusters[:, 1]]
            )
            meat = (
                _cluster_sum(scores, clusters[:, 0])
                + _cluster_sum(scores, clusters[:, 1])
                - _cluster_sum(scores, joined)
            )
        else:
            raise ValueError("clusters must have one or two columns")
        score_cov = scale * meat / n
    else:
        raise ValueError(
            "cov_type must be 'unadjusted', 'robust', or 'clustered'"
        )
    cov = bread @ score_cov @ bread / n
    return (cov + cov.T) / 2


class _IVModel:
    def __init__(
        self,
        dependent,
        exog,
        endog,
        instruments,
        *,
        weights=None,
    ):
        (
            self._y,
            self._x,
            self._z,
            self._weights,
            self._index,
            self._names,
            self._nexog,
        ) = _iv_data(dependent, exog, endog, instruments, weights)
        self.dependent = dependent
        self.exog = exog
        self.endog = endog
        self.instruments = instruments
        self.weights = weights
        self._formula: str | None = None
        self._formula_data = None
        nendog = self._x.shape[1] - self._nexog
        excluded = self._z.shape[1] - self._nexog
        if excluded < nendog:
            raise ValueError(
                "The number of instruments must be at least as large as "
                "the number of endogenous regressors"
            )
        if np.linalg.matrix_rank(self._x) < self._x.shape[1]:
            raise ValueError("regressors do not have full column rank")
        if np.linalg.matrix_rank(self._z) < self._z.shape[1]:
            raise ValueError("instruments do not have full column rank")

    @classmethod
    def from_formula(cls, formula: str, data, *, weights=None, **kwargs):
        match = re.search(r"\[([^\]]+)\]", formula)
        if match is None or "~" not in match.group(1):
            raise ValueError("IV formula must contain [endogenous ~ instruments]")
        endog_text, instrument_text = [
            part.strip() for part in match.group(1).split("~", 1)
        ]
        outer = (formula[: match.start()] + formula[match.end() :]).strip()
        outer = re.sub(r"\+\s*$", "", outer).strip()
        from formulaic import model_matrix

        y, exog = model_matrix(outer, data, output="pandas")
        endog = model_matrix("0 + " + endog_text, data, output="pandas")
        instruments = model_matrix(
            "0 + " + instrument_text, data, output="pandas"
        )
        obj = cls(y, exog, endog, instruments, weights=weights, **kwargs)
        obj._formula = formula
        obj._formula_data = data
        return obj

    def predict(self, params, *, exog=None, endog=None, data=None):
        if data is not None:
            if self._formula is None:
                raise ValueError("data can only be used with from_formula")
            match = re.search(r"\[([^\]]+)\]", self._formula)
            assert match is not None
            endog_text = match.group(1).split("~", 1)[0].strip()
            outer = re.sub(r"\[([^\]]+)\]", "", self._formula)
            outer = re.sub(r"\+\s*$", "", outer).strip()
            from formulaic import model_matrix

            _, exog = model_matrix(outer, data, output="pandas")
            endog = model_matrix("0 + " + endog_text, data, output="pandas")
        if exog is None and endog is None:
            x, index = self._x, self._index
        else:
            ef = frame(exog, "exog") if exog is not None else pd.DataFrame()
            df = frame(endog, "endog") if endog is not None else pd.DataFrame(index=ef.index)
            x = np.ascontiguousarray(np.column_stack([ef, df]), dtype=float)
            index = ef.index if len(ef) else df.index
        values = native_predict(x, np.asarray(params, dtype=float))
        return pd.DataFrame(values, index=index, columns=["predictions"])

    def _fit(self, *, cov_type, debiased, kappa, **cov_config):
        root_w = np.sqrt(self._weights)
        wx = np.ascontiguousarray(self._x * root_w[:, None])
        wy = np.ascontiguousarray(self._y * root_w)
        wz = np.ascontiguousarray(self._z * root_w[:, None])
        if kappa == 1.0:
            beta = native_iv(wx, wy, wz)
        else:
            pzx = wz @ np.linalg.lstsq(wz, wx, rcond=None)[0]
            a = (1.0 - kappa) * (wx.T @ wx) + kappa * (wx.T @ pzx)
            rhs = (1.0 - kappa) * (wx.T @ wy) + kappa * (
                wx.T @ (wz @ np.linalg.lstsq(wz, wy, rcond=None)[0])
            )
            beta = np.linalg.solve(a, rhs)
        cov = _iv_cov(wx, wy, wz, beta, kappa, cov_type, debiased, cov_config)
        fitted = native_predict(self._x, beta)
        residuals = self._y - fitted
        weps = wy - wx @ beta
        constants = np.flatnonzero(np.ptp(self._x, axis=0) < 1e-14)
        if len(constants):
            constant = wx[:, constants[0]]
            centered_y = wy - constant * (
                float(constant @ wy) / float(constant @ constant)
            )
            total_ss = float(centered_y @ centered_y)
        else:
            total_ss = float(wy @ wy)
        return RegressionResults(
            self,
            beta,
            cov,
            residuals,
            fitted,
            self._index,
            self._names,
            transformed_y=wy,
            transformed_residuals=weps,
            df_model=wx.shape[1],
            debiased=debiased,
            kappa=kappa,
            total_ss=total_ss,
        )


class IV2SLS(_IVModel):
    def fit(
        self,
        *,
        cov_type: str = "robust",
        debiased: bool = False,
        **cov_config,
    ):
        return self._fit(
            cov_type=cov_type, debiased=debiased, kappa=1.0, **cov_config
        )


class IVLIML(_IVModel):
    def __init__(
        self,
        dependent,
        exog,
        endog,
        instruments,
        *,
        weights=None,
        fuller=0,
        kappa=None,
    ):
        super().__init__(
            dependent, exog, endog, instruments, weights=weights
        )
        self.fuller = float(fuller)
        self._user_kappa = None if kappa is None else float(kappa)

    def _estimate_kappa(self, wx, wy, wz):
        if self._user_kappa is not None:
            return self._user_kappa
        endogenous = wx[:, self._nexog :]
        e = np.column_stack([wy, endogenous])
        ez = e - wz @ np.linalg.lstsq(wz, e, rcond=None)[0]
        exog = wx[:, : self._nexog]
        if exog.shape[1]:
            ex1 = e - exog @ np.linalg.lstsq(exog, e, rcond=None)[0]
        else:
            ex1 = e
        try:
            from scipy.linalg import eigvalsh

            value = float(eigvalsh(ex1.T @ ex1, ez.T @ ez).min())
        except ImportError:
            value = float(
                np.linalg.eigvals(
                    np.linalg.solve(ez.T @ ez, ex1.T @ ex1)
                ).real.min()
            )
        if self.fuller:
            value -= self.fuller / (len(wy) - wz.shape[1])
        return value

    def fit(
        self,
        *,
        cov_type: str = "robust",
        debiased: bool = False,
        **cov_config,
    ):
        root_w = np.sqrt(self._weights)
        wx = self._x * root_w[:, None]
        wy = self._y * root_w
        wz = self._z * root_w[:, None]
        kappa = self._estimate_kappa(wx, wy, wz)
        return self._fit(
            cov_type=cov_type, debiased=debiased, kappa=kappa, **cov_config
        )
