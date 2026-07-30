from __future__ import annotations

import numpy as np
import pandas as pd

from .._data import formula_matrices, frame, panel_data
from .._lib import demean, first_difference, group_mean, native_ols, native_predict
from .._results import RegressionResults


def _cluster_sum(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    codes, uniques = pd.factorize(groups, sort=False)
    grouped = np.zeros((len(uniques), scores.shape[1]))
    np.add.at(grouped, codes, scores)
    return grouped.T @ grouped


def _panel_cov(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    cov_type: str,
    debiased: bool,
    extra_df: int,
    *,
    entity: np.ndarray,
    time: np.ndarray,
    cov_config: dict,
) -> np.ndarray:
    n, k = x.shape
    bread = np.linalg.inv(x.T @ x)
    eps = y - x @ beta
    effective = n - extra_df - (k if debiased else 0)
    if effective <= 0:
        raise ValueError("insufficient observations after degree-of-freedom adjustment")
    scale = n / effective
    key = cov_type.lower().replace("-", "")
    if key in {"unadjusted", "homoskedastic"}:
        return (eps @ eps / effective) * bread
    scores = x * eps[:, None]
    if key in {"robust", "heteroskedastic"}:
        meat = scores.T @ scores
    elif key in {"clustered", "cluster"}:
        clusters = cov_config.get("clusters")
        if clusters is None:
            selected = []
            if cov_config.get("cluster_entity", False):
                selected.append(entity)
            if cov_config.get("cluster_time", False):
                selected.append(time)
            clusters = np.column_stack(selected) if selected else np.arange(n)
        clusters = np.asarray(clusters)
        if clusters.ndim == 1:
            meat = _cluster_sum(scores, clusters)
            cluster_columns = [clusters]
        elif clusters.ndim == 2 and clusters.shape[1] <= 2:
            cluster_columns = [clusters[:, i] for i in range(clusters.shape[1])]
            if len(cluster_columns) == 1:
                meat = _cluster_sum(scores, cluster_columns[0])
            else:
                if cov_config.get("group_debias", False):
                    raise NotImplementedError(
                        "group_debias is not covered for two-way clustering"
                    )
                joined = pd.MultiIndex.from_arrays(
                    [cluster_columns[0], cluster_columns[1]]
                )
                meat = (
                    _cluster_sum(scores, cluster_columns[0])
                    + _cluster_sum(scores, cluster_columns[1])
                    - _cluster_sum(scores, joined)
                )
        else:
            raise ValueError("Only one- or two-way clustering is supported")
        if cov_config.get("group_debias", False):
            # Upstream applies this per component in the two-way case. This
            # common one-way factor is exact for one-way clustering.
            g = len(np.unique(cluster_columns[0]))
            if g > 1:
                meat *= (g / (g - 1)) * ((n - 1) / n)
    else:
        raise ValueError(
            "cov_type must be 'unadjusted', 'robust', or 'clustered'"
        )
    cov = scale * bread @ meat @ bread
    return (cov + cov.T) / 2


class _PanelModel:
    def __init__(self, dependent, exog, *, weights=None, check_rank: bool = True):
        (
            self._y,
            self._x,
            self._weights,
            self._entity,
            self._time,
            self._index,
            self._names,
        ) = panel_data(dependent, exog, weights)
        self.dependent = dependent
        self.exog = exog
        self.weights = weights
        self.check_rank = bool(check_rank)
        self._formula: str | None = None
        if self._y.ndim != 1:
            raise ValueError("dependent must contain exactly one variable")
        if self.check_rank and np.linalg.matrix_rank(self._x) < self._x.shape[1]:
            raise ValueError("exog does not have full column rank")

    @classmethod
    def from_formula(cls, formula: str, data, *, weights=None, **kwargs):
        y, x, entity_effects, time_effects = formula_matrices(
            formula, data, panel=True
        )
        if cls is PanelOLS:
            kwargs.setdefault("entity_effects", entity_effects)
            kwargs.setdefault("time_effects", time_effects)
        obj = cls(y, x, weights=weights, **kwargs)
        obj._formula = formula
        return obj

    def predict(self, params, *, exog=None, data=None):
        if data is not None:
            if self._formula is None:
                raise ValueError("data can only be used with a model created from_formula")
            _, exog, _, _ = formula_matrices(self._formula, data, panel=True)
        if exog is None:
            x = self._x
            index = self._index
        else:
            xf = frame(exog, "Exog")
            x = np.ascontiguousarray(xf.to_numpy(dtype=np.float64))
            index = xf.index
        values = native_predict(x, np.asarray(params, dtype=np.float64))
        return pd.DataFrame(values, index=index, columns=["predictions"])

    def _finish(
        self,
        beta,
        wx,
        wy,
        *,
        cov_type,
        debiased,
        extra_df,
        cov_config,
        df_model,
        residuals=None,
        fitted=None,
        index=None,
        entity=None,
        time=None,
        effects=None,
        total_ss_override=None,
    ):
        transformed_residuals = wy - wx @ beta
        entity = self._entity if entity is None else entity
        time = self._time if time is None else time
        cov = _panel_cov(
            wx, wy, beta, cov_type, debiased, extra_df,
            entity=entity, time=time, cov_config=cov_config,
        )
        if fitted is None:
            fitted = native_predict(self._x, beta)
        if residuals is None:
            residuals = self._y - fitted
        constant_columns = np.flatnonzero(np.ptp(self._x, axis=0) < 1e-14)
        if total_ss_override is not None:
            total_ss = float(total_ss_override)
        elif len(constant_columns) and wx.shape[1] == self._x.shape[1]:
            constant = wx[:, constant_columns[0]]
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
            self._index if index is None else index,
            self._names,
            transformed_y=wy,
            transformed_residuals=transformed_residuals,
            df_model=df_model,
            debiased=debiased,
            effects=effects,
            total_ss=total_ss,
        )


class PooledOLS(_PanelModel):
    def fit(
        self,
        *,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        **cov_config,
    ):
        root_w = np.sqrt(self._weights)
        wx = np.ascontiguousarray(self._x * root_w[:, None])
        wy = np.ascontiguousarray(self._y * root_w)
        beta, _ = native_ols(wx, wy)
        return self._finish(
            beta, wx, wy, cov_type=cov_type, debiased=debiased, extra_df=0,
            cov_config=cov_config, df_model=self._x.shape[1],
        )


class PanelOLS(_PanelModel):
    def __init__(
        self,
        dependent,
        exog,
        *,
        weights=None,
        entity_effects: bool = False,
        time_effects: bool = False,
        other_effects=None,
        singletons: bool = True,
        drop_absorbed: bool = False,
        check_rank: bool = True,
    ):
        if other_effects is not None:
            raise NotImplementedError("other_effects are not covered")
        super().__init__(dependent, exog, weights=weights, check_rank=check_rank)
        self.entity_effects = bool(entity_effects)
        self.time_effects = bool(time_effects)
        self.other_effects = other_effects
        self.singletons = bool(singletons)
        self.drop_absorbed = bool(drop_absorbed)

    def fit(
        self,
        *,
        use_lsdv: bool = False,
        use_lsmr: bool = False,
        low_memory: bool | None = None,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        auto_df: bool = True,
        count_effects: bool = True,
        **cov_config,
    ):
        if use_lsdv or use_lsmr:
            raise NotImplementedError("dummy-variable and LSMR paths are not covered")
        if (
            self.entity_effects
            and np.any(np.bincount(self._entity) == 1)
        ) or (
            self.time_effects
            and np.any(np.bincount(self._time) == 1)
        ):
            raise NotImplementedError("singleton effect removal is not covered")
        root_w = np.sqrt(self._weights)
        combined = np.column_stack([self._y, self._x])
        entities = int(self._entity.max()) + 1
        periods = int(self._time.max()) + 1
        if self.entity_effects and self.time_effects:
            transformed = demean(
                combined, self._weights, self._entity, entities,
                self._time, periods,
            )
        elif self.entity_effects:
            transformed = demean(
                combined, self._weights, self._entity, entities
            )
        elif self.time_effects:
            transformed = demean(
                combined, self._weights, self._time, periods
            )
        else:
            transformed = np.ascontiguousarray(combined * root_w[:, None])
        wy, wx = transformed[:, 0], np.ascontiguousarray(transformed[:, 1:])

        has_constant = np.any(np.ptp(self._x, axis=0) < 1e-14)
        if has_constant and (self.entity_effects or self.time_effects):
            grand = np.average(combined, axis=0, weights=self._weights)
            wy = np.ascontiguousarray(wy + root_w * grand[0])
            wx = np.ascontiguousarray(wx + root_w[:, None] * grand[None, 1:])
        nonzero = np.linalg.norm(wx, axis=0) > np.finfo(float).eps * len(wx)
        if not np.all(nonzero):
            if not self.drop_absorbed:
                raise ValueError("one or more exogenous variables are fully absorbed")
            wx = np.ascontiguousarray(wx[:, nonzero])
            self._x = np.ascontiguousarray(self._x[:, nonzero])
            self._names = [n for n, keep in zip(self._names, nonzero) if keep]
        beta, _ = native_ols(wx, wy)
        effects_df = 0
        drop_first = has_constant
        if self.entity_effects:
            effects_df += entities - int(drop_first)
            drop_first = True
        if self.time_effects:
            effects_df += periods - int(drop_first)

        extra_df = effects_df if count_effects else 0
        if auto_df and cov_type.lower() in {"clustered", "cluster"}:
            if (
                self.entity_effects
                and not self.time_effects
                and cov_config.get("cluster_entity", False)
            ):
                extra_df = 0
            if (
                self.time_effects
                and not self.entity_effects
                and cov_config.get("cluster_time", False)
            ):
                extra_df = 0
        fitted = native_predict(self._x, beta)
        idio = (wy - wx @ beta) / root_w
        effects = self._y - fitted - idio
        return self._finish(
            beta, wx, wy, cov_type=cov_type, debiased=debiased,
            extra_df=extra_df, cov_config=cov_config,
            df_model=wx.shape[1] + effects_df, residuals=idio, fitted=fitted,
            effects=effects,
        )


class BetweenOLS(_PanelModel):
    def fit(
        self,
        *,
        reweight: bool = False,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        **cov_config,
    ):
        groups = int(self._entity.max()) + 1
        combined = np.column_stack([self._y, self._x])
        means, sums = group_mean(combined, self._weights, self._entity, groups)
        if reweight:
            scale = np.sqrt(sums / sums.mean())
            means = means * scale[:, None]
        wy, wx = means[:, 0], np.ascontiguousarray(means[:, 1:])
        beta, _ = native_ols(wx, wy)
        labels = pd.Index(pd.unique(self._index.get_level_values(0))) \
            if isinstance(self._index, pd.MultiIndex) else pd.RangeIndex(groups)
        index = pd.MultiIndex.from_arrays(
            [labels, np.zeros(groups, dtype=int)], names=["entity", "time"]
        )
        fitted = native_predict(wx, beta)
        residuals = wy - fitted
        codes = np.arange(groups, dtype=np.int64)
        return self._finish(
            beta, wx, wy, cov_type=cov_type, debiased=debiased, extra_df=0,
            cov_config=cov_config, df_model=wx.shape[1], residuals=residuals,
            fitted=fitted, index=index, entity=codes, time=np.zeros(groups, dtype=np.int64),
        )


class FirstDifferenceOLS(_PanelModel):
    def __init__(
        self, dependent, exog, *, weights=None, check_rank: bool = True
    ):
        super().__init__(
            dependent, exog, weights=weights, check_rank=check_rank
        )
        if np.any(np.ptp(self._x, axis=0) < 1e-14):
            raise ValueError("Constants are not allowed in first difference regressions")

    def fit(
        self,
        *,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        **cov_config,
    ):
        combined = np.column_stack([self._y, self._x])
        # panel_data sorts the index before factorizing these codes.
        ordered_entity = self._entity
        ordered_time = self._time
        adjacent = (
            (ordered_entity[1:] == ordered_entity[:-1])
            & (ordered_time[1:] == ordered_time[:-1] + 1)
        )
        right = np.flatnonzero(adjacent).astype(np.int64, copy=False) + 1
        diff = first_difference(combined, right)
        # A differenced observation has harmonic-combination precision.
        keep = right
        current_weight = self._weights[keep]
        previous_weight = self._weights[right - 1]
        effective = (
            current_weight * previous_weight
            / (current_weight + previous_weight)
        )
        effective /= effective.mean()
        root_w = np.sqrt(effective)
        wy = np.ascontiguousarray(diff[:, 0] * root_w)
        wx = np.ascontiguousarray(diff[:, 1:] * root_w[:, None])
        beta, _ = native_ols(wx, wy)
        fitted = native_predict(diff[:, 1:], beta)
        residuals = diff[:, 0] - fitted
        return self._finish(
            beta, wx, wy, cov_type=cov_type, debiased=debiased, extra_df=0,
            cov_config=cov_config, df_model=wx.shape[1], residuals=residuals,
            fitted=fitted, index=self._index[keep], entity=self._entity[keep],
            time=self._time[keep],
        )


class RandomEffects(_PanelModel):
    def fit(
        self,
        *,
        small_sample: bool = False,
        cov_type: str = "unadjusted",
        debiased: bool = True,
        **cov_config,
    ):
        if small_sample:
            raise NotImplementedError(
                "small-sample variance adjustment is not covered"
            )
        groups = int(self._entity.max()) + 1
        combined = np.column_stack([self._y, self._x])
        within = demean(combined, self._weights, self._entity, groups)
        ew_y, ew_x = within[:, 0], np.ascontiguousarray(within[:, 1:])
        has_constant = np.any(np.ptp(self._x, axis=0) < 1e-14)
        if has_constant:
            root_w = np.sqrt(self._weights)
            grand = np.average(combined, axis=0, weights=self._weights)
            ew_y = np.ascontiguousarray(ew_y + root_w * grand[0])
            ew_x = np.ascontiguousarray(
                ew_x + root_w[:, None] * grand[None, 1:]
            )
        beta_within, _ = native_ols(ew_x, ew_y)
        eps = ew_y - ew_x @ beta_within
        n, k = ew_x.shape
        sigma2_e = float(eps @ eps) / max(n - k - groups + 1, 1)

        means, sums = group_mean(
            combined, self._weights, self._entity, groups
        )
        beta_between, _ = native_ols(
            np.ascontiguousarray(means[:, 1:]), means[:, 0]
        )
        eps_bar = means[:, 0] - means[:, 1:] @ beta_between
        counts = np.bincount(self._entity, minlength=groups).astype(float)
        tbar = groups / np.sum(1.0 / counts)
        sigma2_u = max(
            0.0, float(eps_bar @ eps_bar) / max(groups - k, 1) - sigma2_e / tbar
        )
        theta_g = 1.0 - np.sqrt(
            sigma2_e / (counts * sigma2_u + sigma2_e)
        )
        theta = theta_g[self._entity]
        root_w = np.sqrt(self._weights)
        quasi = combined * root_w[:, None] - theta[:, None] * means[self._entity]
        wy, wx = quasi[:, 0], np.ascontiguousarray(quasi[:, 1:])
        beta, _ = native_ols(wx, wy)
        fitted = native_predict(self._x, beta)
        residuals = self._y - fitted
        if has_constant:
            mean_component = root_w * (
                float(root_w @ wy) / float(root_w @ root_w)
            )
            total_ss = float((wy - mean_component) @ (wy - mean_component))
        else:
            total_ss = float(wy @ wy)
        result = self._finish(
            beta, wx, wy, cov_type=cov_type, debiased=debiased, extra_df=0,
            cov_config=cov_config, df_model=k, residuals=residuals, fitted=fitted,
            total_ss_override=total_ss,
        )
        result.theta = pd.DataFrame(
            theta_g,
            index=pd.Index(pd.unique(self._index.get_level_values(0))),
            columns=["theta"],
        )
        result.variance_decomposition = pd.Series(
            {
                "Effects": sigma2_u,
                "Residual": sigma2_e,
                "Percent due to Effects": sigma2_u / (sigma2_u + sigma2_e),
            }
        )
        return result
