from __future__ import annotations

import re

import numpy as np
import pandas as pd


def frame(value, prefix: str) -> pd.DataFrame:
    if isinstance(value, pd.Series):
        return value.to_frame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    arr = np.asarray(value)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{prefix} must be one- or two-dimensional")
    return pd.DataFrame(arr, columns=[f"{prefix}.{i}" for i in range(arr.shape[1])])


def panel_data(dependent, exog, weights=None):
    y = frame(dependent, "dependent")
    x = frame(exog, "Exog")
    if len(y) != len(x):
        raise ValueError("dependent and exog must have the same number of observations")
    if not y.index.equals(x.index):
        x = x.reindex(y.index)
    if weights is None:
        w = pd.Series(1.0, index=y.index)
    else:
        wf = frame(weights, "weight")
        if wf.shape[1] != 1:
            raise ValueError("weights must have one column")
        w = wf.iloc[:, 0].reindex(y.index)
    combined = pd.concat([y, x, w.rename("__weight__")], axis=1)
    combined = combined.dropna()
    if (combined["__weight__"] <= 0).any():
        raise ValueError("weights must be strictly positive")
    if isinstance(combined.index, pd.MultiIndex) and combined.index.nlevels >= 2:
        combined = combined.sort_index()
        entity, _ = pd.factorize(combined.index.get_level_values(0), sort=False)
        time, _ = pd.factorize(combined.index.get_level_values(1), sort=False)
    else:
        entity = np.zeros(len(combined), dtype=np.int64)
        time = np.arange(len(combined), dtype=np.int64)
    yv = np.ascontiguousarray(combined.iloc[:, 0].to_numpy(dtype=np.float64))
    xv = np.ascontiguousarray(
        combined.iloc[:, 1:-1].to_numpy(dtype=np.float64)
    )
    wv = np.array(
        combined.iloc[:, -1].to_numpy(dtype=np.float64), order="C", copy=True
    )
    wv /= wv.mean()
    return (
        yv,
        xv,
        wv,
        np.ascontiguousarray(entity, dtype=np.int64),
        np.ascontiguousarray(time, dtype=np.int64),
        combined.index,
        list(x.columns.astype(str)),
    )


def formula_matrices(formula: str, data, *, panel: bool):
    entity_effects = bool(re.search(r"\bEntityEffects\b", formula))
    time_effects = bool(re.search(r"\bTimeEffects\b", formula))
    clean = re.sub(r"\s*\+\s*EntityEffects\b", "", formula)
    clean = re.sub(r"\s*\+\s*TimeEffects\b", "", clean)
    try:
        from formulaic import model_matrix
    except ImportError as exc:
        raise ImportError("formula use requires formulaic") from exc
    y, x = model_matrix(clean, data, output="pandas")
    return y, x, entity_effects, time_effects
