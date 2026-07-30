from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WaldTestStatistic:
    stat: float
    pval: float
    df: int
    null: str = "All coefficients are zero"


class RegressionResults:
    def __init__(
        self,
        model,
        params: np.ndarray,
        cov: np.ndarray,
        residuals: np.ndarray,
        fitted: np.ndarray,
        index,
        names: list[str],
        *,
        transformed_y: np.ndarray,
        transformed_residuals: np.ndarray,
        df_model: int,
        debiased: bool,
        kappa: float = 0.0,
        effects: np.ndarray | None = None,
        total_ss: float | None = None,
    ):
        self.model = model
        self.params = pd.Series(params, index=names, name="parameter")
        self.cov = pd.DataFrame(cov, index=names, columns=names)
        # A negative diagonal indicates an indefinite covariance estimate
        # (possible with finite-sample multiway clustering).  Preserve that as
        # NaN instead of silently presenting a zero standard error.
        with np.errstate(invalid="ignore"):
            std_errors = np.sqrt(np.diag(cov))
        self.std_errors = pd.Series(std_errors, index=names, name="std_error")
        self.tstats = self.params / self.std_errors
        self.tstats.name = "tstat"
        self.nobs = len(residuals)
        self.df_model = int(df_model)
        self.df_resid = self.nobs - self.df_model
        self.debiased = bool(debiased)
        self.resids = pd.Series(residuals, index=index, name="residual")
        self.fitted_values = pd.DataFrame(
            fitted, index=index, columns=["fitted_values"]
        )
        self.idiosyncratic = pd.DataFrame(
            residuals, index=index, columns=["idiosyncratic"]
        )
        self.estimated_effects = pd.DataFrame(
            np.nan if effects is None else effects,
            index=index,
            columns=["estimated_effects"],
        )
        self._kappa = float(kappa)
        self._rss = float(transformed_residuals @ transformed_residuals)
        if total_ss is None:
            centered = transformed_y - transformed_y.mean()
            total_ss = float(centered @ centered)
        self.total_ss = float(total_ss)
        self.resid_ss = self._rss
        self.model_ss = self.total_ss - self.resid_ss
        self.rsquared = (
            1.0 - self.resid_ss / self.total_ss if self.total_ss > 0 else 0.0
        )
        self.s2 = self.resid_ss / max(self.df_resid, 1)
        try:
            from scipy.stats import norm, t

            dist = t(df=max(self.df_resid, 1)) if debiased else norm()
            self.pvalues = pd.Series(
                2 * dist.sf(np.abs(self.tstats)), index=names, name="pvalue"
            )
        except ImportError:
            self.pvalues = pd.Series(
                [math.erfc(abs(v) / math.sqrt(2.0)) for v in self.tstats],
                index=names,
                name="pvalue",
            )
        stat = float(np.sum(np.square(self.tstats.to_numpy())))
        self.f_statistic = WaldTestStatistic(stat, float("nan"), len(names))

    @property
    def kappa(self) -> float:
        return self._kappa

    @property
    def rsquared_within(self) -> float:
        return self.rsquared

    @property
    def rsquared_overall(self) -> float:
        y = self.model._y
        e = self.resids.to_numpy()
        centered = y - y.mean()
        den = centered @ centered
        return 1.0 - float(e @ e) / den if den else 0.0

    @property
    def summary(self) -> str:
        table = pd.DataFrame(
            {
                "Parameter": self.params,
                "Std. Err.": self.std_errors,
                "T-stat": self.tstats,
                "P-value": self.pvalues,
            }
        )
        return (
            f"{self.model.__class__.__name__} Estimation Summary\n"
            f"No. Observations: {self.nobs}\nR-squared: {self.rsquared:.6f}\n\n"
            + table.to_string()
        )

    def predict(
        self,
        exog=None,
        *,
        data=None,
        fitted: bool = True,
        effects: bool = False,
        idiosyncratic: bool = False,
        missing: bool = False,
    ):
        if exog is not None or data is not None:
            return self.model.predict(self.params, exog=exog, data=data)
        pieces = []
        if fitted:
            pieces.append(self.fitted_values)
        if effects:
            pieces.append(self.estimated_effects)
        if idiosyncratic:
            pieces.append(self.idiosyncratic)
        return pd.concat(pieces, axis=1)

    def __str__(self) -> str:
        return self.summary
