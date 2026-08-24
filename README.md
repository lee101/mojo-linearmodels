# mojo-linearmodels

Panel-data and instrumental-variable estimators implemented in
[Mojo](https://www.modular.com/mojo) and exposed to Python. The covered classes
provide a deliberately limited, parity-tested subset of the corresponding
[`linearmodels`](https://bashtage.github.io/linearmodels/) interfaces. They
accept pandas or NumPy inputs, weights, formulas, supported covariance options,
and expose familiar core result attributes.

The Python import is deliberately `mojo_linearmodels`, so this package and the
upstream package can be installed together for parity testing:

```python
import numpy as np
import pandas as pd
from mojo_linearmodels.panel import PanelOLS

rng = np.random.default_rng(0)
index = pd.MultiIndex.from_product(
    [["a", "b", "c"], range(8)], names=["firm", "year"]
)
x = rng.normal(size=len(index))
firm_effect = np.repeat([0.5, -0.2, 0.8], 8)
data = pd.DataFrame(
    {"y": 1.5 * x + firm_effect + rng.normal(scale=0.1, size=len(index)), "x": x},
    index=index,
)

result = PanelOLS.from_formula("y ~ x + EntityEffects", data).fit(
    cov_type="clustered", cluster_entity=True
)
print(result.params)
```

## Coverage

| area | covered |
| --- | --- |
| Panel models | `PooledOLS`, `PanelOLS`, `BetweenOLS`, `FirstDifferenceOLS`, `RandomEffects` |
| Fixed effects | Entity, time, and combined entity-time absorption; balanced and unbalanced panels |
| IV models | `IV2SLS`, `IVLIML`, Fuller adjustment, user-specified LIML kappa |
| Inputs | NumPy, pandas, missing-row removal, positive observation weights |
| Covariance | Homoskedastic and robust for every listed estimator; one- and two-way clustered for pooled/fixed-effects panel OLS and IV |
| Interface | Direct constructors, `from_formula`, `fit`, model/result `predict`, parameters, covariance, standard errors, statistics, residuals, fitted values, and R-squared |

The numerical parity suite compares all covered estimators against the real
PyPI/conda `linearmodels` package. It includes weighted and unbalanced panels,
robust and clustered standard errors, formula models, LIML, Fuller LIML,
user-specified kappa, SIMD tail lengths, and native-boundary validation.

Not covered are `other_effects`, explicit LSDV/LSMR fitting, singleton removal,
Driscoll-Kraay or kernel HAC covariance, clustered covariance for between,
first-difference, or random-effects models, two-way clustered `group_debias`,
the random-effects `small_sample` adjustment, IVGMM/CUE, absorbing least
squares, system regressions, asset-pricing models, first-stage diagnostic
result objects, and the full upstream summary/test-statistic surface. Cases
accepted by the API that could otherwise produce a misleading answer raise a
clear error.

## Install and run

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` creates `dist/libmojo-linearmodels.so`. The Python loader also
rebuilds a missing or stale library. A prebuilt library can be selected with
`MOJO_LINEARMODELS_LIB=/path/to/libmojo-linearmodels.so`.

## Benchmarks

These are full `.fit()` timings from `pixi run bench`, including data
adaptation, covariance estimation, and result construction for both
implementations. Each value is the best of three runs after one warm-up.

Machine: Intel Xeon E5-2697 v4 at 2.30 GHz, Python 3.13.14.

| case | mojo-linearmodels | linearmodels | result |
| --- | ---: | ---: | ---: |
| PooledOLS.fit robust (200k x 9) | 271.8 ms | 1620.5 ms | 5.96x faster |
| PanelOLS entity FE (160k x 7) | 402.2 ms | 1283.3 ms | 3.19x faster |
| BetweenOLS.fit (500k x 6) | 465.3 ms | 3086.0 ms | 6.63x faster |
| FirstDifferenceOLS.fit (240k x 6) | 296.1 ms | 1683.0 ms | 5.68x faster |
| RandomEffects.fit (160k x 7) | 590.2 ms | 1503.7 ms | 2.55x faster |
| IV2SLS.fit robust (250k, 7 regressors) | 298.6 ms | 2308.8 ms | 7.73x faster |

These results describe this machine and workload, not a universal speedup.
Upstream can win on different matrix shapes or BLAS/thread configurations.

No GPU path is included. The profiled work is dominated by short-width group
transforms, differences, predictions, and small-matrix cross-products with
less than roughly two floating-point operations per byte moved. These kernels
are bandwidth-bound and do not justify host/device transfer overhead.

## How it works

All native routines live in one Mojo compilation unit. Python owns input,
output, and scratch arrays and calls a small C ABI with `ctypes`. Buffers cross
the boundary as integer addresses and are reconstructed as
`UnsafePointer[..., AnyOrigin[mut=True]]` inside non-parametric
`@export(...)` functions.

Arrays are C-contiguous `float64` in row-major order; category and panel codes
are contiguous `int64`. Mojo performs SIMD rank-one Gram updates, cross
products, Cholesky solves, weighted one- and two-way absorption, group means,
first differences, 2SLS projections, and prediction. Large independent row
and sorted-group transforms use thresholded CPU parallelism. Python handles
pandas index alignment, formulas, covariance assembly, and result objects.
Mojo does not allocate array buffers: every buffer lifetime remains owned by
NumPy.

## License

MIT
