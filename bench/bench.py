"""Benchmarks against linearmodels on identical pandas/NumPy inputs."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"
    ),
)

from linearmodels.iv import IV2SLS as UpIV2SLS  # noqa: E402
from linearmodels.panel import (  # noqa: E402
    BetweenOLS as UpBetweenOLS,
    FirstDifferenceOLS as UpFirstDifferenceOLS,
    PanelOLS as UpPanelOLS,
    PooledOLS as UpPooledOLS,
    RandomEffects as UpRandomEffects,
)
from mojo_linearmodels.iv import IV2SLS  # noqa: E402
from mojo_linearmodels.panel import (  # noqa: E402
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    PooledOLS,
    RandomEffects,
)


def best_time(fn, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def panel_data(entities, periods, variables, seed=0, constant=True):
    rng = np.random.default_rng(seed)
    n = entities * periods
    index = pd.MultiIndex.from_product(
        [range(entities), range(periods)], names=["entity", "time"]
    )
    values = rng.normal(size=(n, variables))
    columns = [f"x{i}" for i in range(variables)]
    x = pd.DataFrame(values, index=index, columns=columns)
    if constant:
        x.insert(0, "const", 1.0)
    beta = rng.normal(size=x.shape[1])
    effects = np.repeat(rng.normal(size=entities), periods)
    y = pd.Series(
        x.to_numpy() @ beta + effects + rng.normal(size=n),
        index=index,
        name="y",
    )
    return y, x


def cases():
    y_pool, x_pool = panel_data(50_000, 4, 8, seed=1)
    yield (
        "PooledOLS.fit robust (200k x 9)",
        lambda: PooledOLS(y_pool, x_pool).fit(cov_type="robust"),
        lambda: UpPooledOLS(y_pool, x_pool).fit(cov_type="robust"),
    )

    y_fe, x_fe = panel_data(20_000, 8, 6, seed=2)
    yield (
        "PanelOLS entity FE (160k x 7)",
        lambda: PanelOLS(y_fe, x_fe, entity_effects=True).fit(),
        lambda: UpPanelOLS(y_fe, x_fe, entity_effects=True).fit(),
    )

    y_between, x_between = panel_data(100_000, 5, 5, seed=3)
    yield (
        "BetweenOLS.fit (500k x 6)",
        lambda: BetweenOLS(y_between, x_between).fit(),
        lambda: UpBetweenOLS(y_between, x_between).fit(),
    )

    y_fd, x_fd = panel_data(40_000, 6, 6, seed=4, constant=False)
    yield (
        "FirstDifferenceOLS.fit (240k x 6)",
        lambda: FirstDifferenceOLS(y_fd, x_fd).fit(),
        lambda: UpFirstDifferenceOLS(y_fd, x_fd).fit(),
    )

    y_re, x_re = panel_data(20_000, 8, 6, seed=5)
    yield (
        "RandomEffects.fit (160k x 7)",
        lambda: RandomEffects(y_re, x_re).fit(),
        lambda: UpRandomEffects(y_re, x_re).fit(),
    )

    rng = np.random.default_rng(6)
    n, exog_count, instrument_count = 250_000, 5, 4
    exog = np.column_stack([np.ones(n), rng.normal(size=(n, exog_count))])
    instruments = rng.normal(size=(n, instrument_count))
    confounder = rng.normal(size=n)
    endog = (
        instruments @ rng.normal(size=instrument_count)
        + confounder
        + rng.normal(size=n)
    )
    y_iv = exog @ rng.normal(size=exog.shape[1]) + 1.8 * endog + confounder
    yield (
        "IV2SLS.fit robust (250k, 7 regressors)",
        lambda: IV2SLS(y_iv, exog, endog, instruments).fit(),
        lambda: UpIV2SLS(y_iv, exog, endog, instruments).fit(),
    )


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    print(f"Machine: {cpu_name()}; Python {platform.python_version()}")
    print("| case | mojo-linearmodels | linearmodels | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, upstream in cases():
        ours()
        upstream()
        a = best_time(ours)
        b = best_time(upstream)
        if a <= b:
            result = f"{b / a:.2f}x faster"
        else:
            result = f"{a / b:.2f}x slower"
        print(f"| {name} | {a * 1000:.1f} ms | {b * 1000:.1f} ms | {result} |")


if __name__ == "__main__":
    main()
