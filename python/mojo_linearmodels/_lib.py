from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIB = Path(
    os.environ.get(
        "MOJO_LINEARMODELS_LIB", ROOT / "dist/libmojo-linearmodels.so"
    )
)
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mlm_group_demean": ([I] * 9, None),
    "mlm_two_way_demean": ([I] * 11 + [I, F], I),
    "mlm_group_mean": ([I] * 8, None),
    "mlm_first_difference": ([I] * 5, None),
    "mlm_ols": ([I] * 6, I),
    "mlm_cross": ([I] * 6, None),
    "mlm_predict": ([I] * 5, None),
    "mlm_iv2sls": ([I] * 8, I),
}

_library: ctypes.CDLL | None = None


def build() -> Path:
    source = ROOT / "src/kernels.mojo"
    if LIB.exists() and LIB.stat().st_mtime >= source.stat().st_mtime:
        return LIB
    proc = subprocess.run(
        ["bash", str(ROOT / "build/build.sh")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=1800,
    )
    if proc.returncode or not LIB.exists():
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return LIB


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(str(build()))
        for name, (args, result) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = args
            fn.restype = result
    return _library


def f64(value, *, copy: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise TypeError("native arrays must contain real numeric values")
    if raw.dtype.kind in "iu" and raw.size:
        # Float64 cannot exactly represent larger integers.  Refuse a silent,
        # potentially model-changing conversion at the FFI boundary.
        limit = 1 << 53
        if np.any(raw > limit) or np.any(raw < -limit):
            raise ValueError("integer values outside the exact float64 range")
    if copy:
        result = np.array(raw, dtype=np.float64, order="C", copy=True)
    else:
        result = np.ascontiguousarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("native arrays must contain only finite values")
    return result


def i64(value) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise TypeError("native index arrays must contain integers")
    if raw.dtype.kind == "u" and raw.size and raw.max() > np.iinfo(np.int64).max:
        raise ValueError("index value exceeds the int64 range")
    return np.ascontiguousarray(raw, dtype=np.int64)


def addr(value: np.ndarray) -> int:
    address = int(value.ctypes.data)
    if value.size and not address:
        raise ValueError("NumPy returned a null pointer for a non-empty array")
    return address


def _matrix(value, name: str) -> np.ndarray:
    result = f64(value)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not result.shape[0] or not result.shape[1]:
        raise ValueError(f"{name} must not be empty")
    return result


def _vector(value, name: str, *, length: int | None = None) -> np.ndarray:
    result = f64(value)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must have length {length}")
    return result


def _codes(value, name: str, length: int, groups: int) -> np.ndarray:
    if groups <= 0:
        raise ValueError("groups must be positive")
    result = i64(value)
    if result.ndim != 1 or len(result) != length:
        raise ValueError(f"{name} must be a one-dimensional array of length {length}")
    if result.size and (result.min() < 0 or result.max() >= groups):
        raise ValueError(f"{name} values must be in [0, groups)")
    return result


def native_ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _matrix(x, "x")
    n, k = x.shape
    y = _vector(y, "y", length=n)
    beta = np.empty(k)
    work = np.empty((k, k))
    ok = lib().mlm_ols(addr(x), addr(y), addr(beta), addr(work), n, k)
    if not ok:
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
    return beta, x.T @ x


def native_predict(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    x = _matrix(x, "x")
    beta = _vector(beta, "beta", length=x.shape[1])
    result = np.empty(x.shape[0])
    lib().mlm_predict(addr(x), addr(beta), addr(result), x.shape[0], x.shape[1])
    return result


def group_mean(
    x: np.ndarray, weights: np.ndarray, codes: np.ndarray, groups: int
) -> tuple[np.ndarray, np.ndarray]:
    x = _matrix(x, "x")
    weights = _vector(weights, "weights", length=x.shape[0])
    codes = _codes(codes, "codes", x.shape[0], groups)
    result = np.empty((groups, x.shape[1]))
    sums = np.empty(groups)
    lib().mlm_group_mean(
        addr(x), addr(weights), addr(codes), addr(result), addr(sums),
        x.shape[0], x.shape[1], groups,
    )
    return result, sums


def demean(
    x: np.ndarray,
    weights: np.ndarray,
    entity: np.ndarray,
    entities: int,
    time: np.ndarray | None = None,
    periods: int = 0,
) -> np.ndarray:
    x = _matrix(x, "x")
    weights = _vector(weights, "weights", length=x.shape[0])
    entity = _codes(entity, "entity", x.shape[0], entities)
    result = np.empty_like(x)
    means = np.empty((max(entities, periods), x.shape[1]))
    sums = np.empty(max(entities, periods))
    if time is None:
        lib().mlm_group_demean(
            addr(x), addr(weights), addr(entity), addr(result), addr(means),
            addr(sums), x.shape[0], x.shape[1], entities,
        )
    else:
        time = _codes(time, "time", x.shape[0], periods)
        iterations = lib().mlm_two_way_demean(
            addr(x), addr(weights), addr(entity), addr(time), addr(result),
            addr(means), addr(sums), x.shape[0], x.shape[1], entities, periods,
            1000, 1e-12,
        )
        if iterations >= 1000:
            raise RuntimeError("two-way fixed-effect absorption did not converge")
    return result


def first_difference(
    x: np.ndarray, right: np.ndarray
) -> np.ndarray:
    x = _matrix(x, "x")
    right = i64(right)
    if right.ndim != 1:
        raise ValueError("right must be a one-dimensional index array")
    if right.size and (right.min() < 1 or right.max() >= x.shape[0]):
        raise ValueError("right indices must be between 1 and len(x) - 1")
    if not len(right):
        return np.empty((0, x.shape[1]), dtype=np.float64)
    result = np.empty((len(right), x.shape[1]))
    lib().mlm_first_difference(
        addr(x), addr(right), addr(result), len(right), x.shape[1]
    )
    return result


def native_iv(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    x = _matrix(x, "x")
    n, k = x.shape
    y = _vector(y, "y", length=n)
    z = _matrix(z, "z")
    if z.shape[0] != n:
        raise ValueError("z must have the same number of rows as x")
    ell = z.shape[1]
    beta = np.empty(k)
    work_size = ell * ell + 2 * ell * k + ell + k * k + max(ell, k)
    work = np.empty(work_size)
    ok = lib().mlm_iv2sls(
        addr(x), addr(y), addr(z), addr(beta), addr(work), n, k, ell
    )
    if not ok:
        ztx = z.T @ x
        beta = np.linalg.lstsq(
            ztx.T @ np.linalg.lstsq(z.T @ z, ztx, rcond=None)[0],
            ztx.T @ np.linalg.lstsq(z.T @ z, z.T @ y, rcond=None)[0],
            rcond=None,
        )[0]
    return beta
