"""Dense panel-data and IV kernels exposed through one C ABI compilation unit."""

from std.math import sqrt
from std.sys.info import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime SORTED_GROUP_ELEMENTS = 262_144


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def dot(a: Ptr, b: Ptr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.unsafe_load[width=W](i) * b.unsafe_load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a[unsafe_offset=i] * b[unsafe_offset=i]
        i += 1
    return total


def axpy(alpha: Float64, x: Ptr, y: Ptr, n: Int):
    var va = SIMD[DType.float64, W](alpha)
    var i = 0
    while i + W <= n:
        y.unsafe_store(
            i, y.unsafe_load[width=W](i) + va * x.unsafe_load[width=W](i)
        )
        i += W
    while i < n:
        y[unsafe_offset=i] += alpha * x[unsafe_offset=i]
        i += 1


def copy_flat(src: Ptr, dst: Ptr, n: Int):
    var i = 0
    while i + W <= n:
        dst.unsafe_store(i, src.unsafe_load[width=W](i))
        i += W
    while i < n:
        dst[unsafe_offset=i] = src[unsafe_offset=i]
        i += 1


def scale_row(data: Ptr, scale: Float64, d: Int):
    var value = SIMD[DType.float64, W](scale)
    var j = 0
    while j + W <= d:
        data.unsafe_store(j, data.unsafe_load[width=W](j) * value)
        j += W
    while j < d:
        data[unsafe_offset=j] *= scale
        j += 1


def scale_rows(data: Ptr, weights: Ptr, n: Int, d: Int):
    for i in range(n):
        scale_row(
            data.unsafe_offset(i * d), sqrt(weights[unsafe_offset=i]), d
        )


def gram(x: Ptr, dst: Ptr, n: Int, d: Int):
    for i in range(d * d):
        dst[unsafe_offset=i] = 0.0
    for r in range(n):
        var row = x.unsafe_offset(r * d)
        for i in range(d):
            var value = row[unsafe_offset=i]
            if value != 0.0:
                axpy(value, row, dst.unsafe_offset(i * d), i + 1)
    for i in range(d):
        for j in range(i + 1, d):
            dst[unsafe_offset=i * d + j] = dst[unsafe_offset=j * d + i]


def cross(a: Ptr, b: Ptr, dst: Ptr, n: Int, da: Int, db: Int):
    for i in range(da * db):
        dst[unsafe_offset=i] = 0.0
    for r in range(n):
        var ar = a.unsafe_offset(r * da)
        var br = b.unsafe_offset(r * db)
        for i in range(da):
            var value = ar[unsafe_offset=i]
            if value != 0.0:
                axpy(value, br, dst.unsafe_offset(i * db), db)


def cholesky(a: Ptr, d: Int) -> Bool:
    for i in range(d):
        for j in range(i + 1):
            var acc = a[unsafe_offset=i * d + j]
            for k in range(j):
                acc -= a[unsafe_offset=i * d + k] * a[unsafe_offset=j * d + k]
            if i == j:
                if acc <= 0.0:
                    return False
                a[unsafe_offset=i * d + i] = sqrt(acc)
            else:
                a[unsafe_offset=i * d + j] = acc / a[unsafe_offset=j * d + j]
    return True


def cholesky_solve(l: Ptr, b: Ptr, d: Int):
    for i in range(d):
        var acc = b[unsafe_offset=i]
        for k in range(i):
            acc -= l[unsafe_offset=i * d + k] * b[unsafe_offset=k]
        b[unsafe_offset=i] = acc / l[unsafe_offset=i * d + i]
    for ri in range(d):
        var i = d - 1 - ri
        var acc = b[unsafe_offset=i]
        for k in range(i + 1, d):
            acc -= l[unsafe_offset=k * d + i] * b[unsafe_offset=k]
        b[unsafe_offset=i] = acc / l[unsafe_offset=i * d + i]


def solve_columns(l: Ptr, b: Ptr, rows: Int, cols: Int, work: Ptr):
    for j in range(cols):
        for i in range(rows):
            work[unsafe_offset=i] = b[unsafe_offset=i * cols + j]
        cholesky_solve(l, work, rows)
        for i in range(rows):
            b[unsafe_offset=i * cols + j] = work[unsafe_offset=i]


def ols(
    x: Ptr, y: Ptr, beta: Ptr, work: Ptr, gram_result: Ptr, n: Int, k: Int
) -> Bool:
    gram(x, work, n, k)
    copy_flat(work, gram_result, k * k)
    for j in range(k):
        var acc = 0.0
        for i in range(n):
            acc += x[unsafe_offset=i * k + j] * y[unsafe_offset=i]
        beta[unsafe_offset=j] = acc
    if not cholesky(work, k):
        return False
    cholesky_solve(work, beta, k)
    return True


def group_center_once(
    data: Ptr, weights: Ptr, codes: IPtr, n: Int, d: Int, groups: Int,
    means: Ptr, sums: Ptr
) -> Float64:
    for g in range(groups):
        sums[unsafe_offset=g] = 0.0
    for i in range(groups * d):
        means[unsafe_offset=i] = 0.0
    for i in range(n):
        var g = Int(codes[unsafe_offset=i])
        var wi = weights[unsafe_offset=i]
        sums[unsafe_offset=g] += wi
        axpy(wi, data.unsafe_offset(i * d), means.unsafe_offset(g * d), d)
    var largest = 0.0
    for g in range(groups):
        if sums[unsafe_offset=g] > 0.0:
            var inv = 1.0 / sums[unsafe_offset=g]
            for j in range(d):
                var value = means[unsafe_offset=g * d + j] * inv
                means[unsafe_offset=g * d + j] = value
                largest = max(largest, abs(value))
    for i in range(n):
        axpy(
            -1.0,
            means.unsafe_offset(Int(codes[unsafe_offset=i]) * d),
            data.unsafe_offset(i * d),
            d,
        )
    return largest


def codes_are_sorted(codes: IPtr, n: Int) -> Bool:
    for i in range(1, n):
        if codes[unsafe_offset=i] < codes[unsafe_offset=i - 1]:
            return False
    return True


def grouped_sorted(
    data: Ptr, weights: Ptr, codes: IPtr, n: Int, d: Int, groups: Int,
    means: Ptr, sums: Ptr, center: Bool
):
    for g in range(groups):
        sums[unsafe_offset=g] = 0.0
    for i in range(groups * d):
        means[unsafe_offset=i] = 0.0

    # Codes are sorted, so whole groups are contiguous: walking them in index
    # order visits every row exactly once and keeps each group's accumulation
    # order identical to the per-group partitioning used before.
    var i = 0
    while i < n:
        var g = Int(codes[unsafe_offset=i])
        var group_end = i + 1
        while (
            group_end < n
            and codes[unsafe_offset=group_end] == codes[unsafe_offset=i]
        ):
            group_end += 1
        for row in range(i, group_end):
            var wi = weights[unsafe_offset=row]
            sums[unsafe_offset=g] += wi
            axpy(
                wi, data.unsafe_offset(row * d),
                means.unsafe_offset(g * d), d,
            )
        scale_row(
            means.unsafe_offset(g * d), 1.0 / sums[unsafe_offset=g], d
        )
        if center:
            for row in range(i, group_end):
                axpy(
                    -1.0, means.unsafe_offset(g * d),
                    data.unsafe_offset(row * d), d,
                )
        i = group_end


@export("mlm_group_demean")
def mlm_group_demean(
    src: Int, weights: Int, codes: Int, dst: Int, means: Int, sums: Int,
    n: Int, d: Int, groups: Int
) abi("C"):
    var x = p(src)
    var w = p(weights)
    var result = p(dst)
    copy_flat(x, result, n * d)
    if (
        n * d >= SORTED_GROUP_ELEMENTS
        and groups > 1
        and codes_are_sorted(ip(codes), n)
    ):
        grouped_sorted(
            result, w, ip(codes), n, d, groups, p(means), p(sums), True
        )
    else:
        _ = group_center_once(
            result, w, ip(codes), n, d, groups, p(means), p(sums)
        )
    scale_rows(result, w, n, d)


@export("mlm_two_way_demean")
def mlm_two_way_demean(
    src: Int, weights: Int, entity: Int, time: Int, dst: Int,
    means: Int, sums: Int, n: Int, d: Int, entities: Int, periods: Int,
    max_iter: Int, tol: Float64
) abi("C") -> Int:
    var x = p(src)
    var w = p(weights)
    var result = p(dst)
    copy_flat(x, result, n * d)
    var iterations = 0
    while iterations < max_iter:
        var a = group_center_once(
            result, w, ip(entity), n, d, entities, p(means), p(sums)
        )
        var b = group_center_once(
            result, w, ip(time), n, d, periods, p(means), p(sums)
        )
        iterations += 1
        if max(a, b) < tol:
            break
    scale_rows(result, w, n, d)
    return iterations


@export("mlm_group_mean")
def mlm_group_mean(
    src: Int, weights: Int, codes: Int, dst: Int, sums: Int,
    n: Int, d: Int, groups: Int
) abi("C"):
    var x = p(src)
    var w = p(weights)
    var result = p(dst)
    if (
        n * d >= SORTED_GROUP_ELEMENTS
        and groups > 1
        and codes_are_sorted(ip(codes), n)
    ):
        grouped_sorted(
            x, w, ip(codes), n, d, groups, result, p(sums), False
        )
        return
    for i in range(groups * d):
        result[unsafe_offset=i] = 0.0
    for g in range(groups):
        p(sums)[unsafe_offset=g] = 0.0
    for i in range(n):
        var g = Int(ip(codes)[unsafe_offset=i])
        var wi = w[unsafe_offset=i]
        p(sums)[unsafe_offset=g] += wi
        axpy(wi, x.unsafe_offset(i * d), result.unsafe_offset(g * d), d)
    for g in range(groups):
        if p(sums)[unsafe_offset=g] > 0.0:
            var inv = 1.0 / p(sums)[unsafe_offset=g]
            scale_row(result.unsafe_offset(g * d), inv, d)


def difference_row(x: Ptr, right: IPtr, result: Ptr, row: Int, d: Int):
    var source_row = Int(right[unsafe_offset=row])
    var current = x.unsafe_offset(source_row * d)
    var previous = current.unsafe_offset(-d)
    var target = result.unsafe_offset(row * d)
    var j = 0
    while j + W <= d:
        target.unsafe_store(
            j,
            current.unsafe_load[width=W](j)
            - previous.unsafe_load[width=W](j),
        )
        j += W
    while j < d:
        target[unsafe_offset=j] = (
            current[unsafe_offset=j] - previous[unsafe_offset=j]
        )
        j += 1


def first_difference(
    src: Int, right_addr: Int, dst: Int, rows: Int, d: Int
):
    var x = p(src)
    var right = ip(right_addr)
    var result = p(dst)
    for row in range(rows):
        difference_row(x, right, result, row, d)


@export("mlm_first_difference")
def mlm_first_difference(
    src: Int, right: Int, dst: Int, rows: Int, d: Int
) abi("C"):
    first_difference(src, right, dst, rows, d)


@export("mlm_ols")
def mlm_ols(
    x: Int, y: Int, beta: Int, work: Int, gram_result: Int, n: Int, k: Int
) abi("C") -> Int:
    return 1 if ols(
        p(x), p(y), p(beta), p(work), p(gram_result), n, k
    ) else 0


@export("mlm_cross")
def mlm_cross(
    a: Int, b: Int, dst: Int, n: Int, da: Int, db: Int
) abi("C"):
    cross(p(a), p(b), p(dst), n, da, db)


@export("mlm_predict")
def mlm_predict(
    x: Int, beta: Int, dst: Int, n: Int, k: Int
) abi("C"):
    var xp = p(x)
    var bp = p(beta)
    var result = p(dst)
    for i in range(n):
        result[unsafe_offset=i] = dot(xp.unsafe_offset(i * k), bp, k)


@export("mlm_quasi_demean")
def mlm_quasi_demean(
    src: Int, weights: Int, codes: Int, means: Int, theta: Int, dst: Int,
    n: Int, d: Int
) abi("C"):
    var xp = p(src)
    var wp = p(weights)
    var cp = ip(codes)
    var mp = p(means)
    var tp = p(theta)
    var result = p(dst)
    for i in range(n):
        var g = Int(cp[unsafe_offset=i])
        var root_w = SIMD[DType.float64, W](sqrt(wp[unsafe_offset=i]))
        var theta_g = SIMD[DType.float64, W](tp[unsafe_offset=g])
        var source = xp.unsafe_offset(i * d)
        var mean = mp.unsafe_offset(g * d)
        var target = result.unsafe_offset(i * d)
        var j = 0
        while j + W <= d:
            target.unsafe_store(
                j,
                source.unsafe_load[width=W](j) * root_w
                - mean.unsafe_load[width=W](j) * theta_g,
            )
            j += W
        while j < d:
            target[unsafe_offset=j] = (
                source[unsafe_offset=j] * sqrt(wp[unsafe_offset=i])
                - mean[unsafe_offset=j] * tp[unsafe_offset=g]
            )
            j += 1


@export("mlm_iv2sls")
def mlm_iv2sls(
    x: Int, y: Int, z: Int, beta: Int, projection: Int, work: Int,
    n: Int, k: Int, instruments: Int
) abi("C") -> Int:
    var xp = p(x)
    var yp = p(y)
    var zp = p(z)
    var wp = p(work)
    var ztz = wp
    var ztx = ztz.unsafe_offset(instruments * instruments)
    var ztx_solved = ztx.unsafe_offset(instruments * k)
    var zty = ztx_solved.unsafe_offset(instruments * k)
    var xpzx = zty.unsafe_offset(instruments)
    var temp = xpzx.unsafe_offset(k * k)

    gram(zp, ztz, n, instruments)
    if not cholesky(ztz, instruments):
        return 0
    cross(zp, xp, ztx, n, instruments, k)
    for j in range(instruments):
        var acc = 0.0
        for i in range(n):
            acc += (
                zp[unsafe_offset=i * instruments + j] * yp[unsafe_offset=i]
            )
        zty[unsafe_offset=j] = acc
    for i in range(instruments * k):
        ztx_solved[unsafe_offset=i] = ztx[unsafe_offset=i]
    solve_columns(ztz, ztx_solved, instruments, k, temp)
    copy_flat(ztx_solved, p(projection), instruments * k)
    cholesky_solve(ztz, zty, instruments)

    for i in range(k):
        for j in range(i + 1):
            var acc = 0.0
            for q in range(instruments):
                acc += (
                    ztx[unsafe_offset=q * k + i]
                    * ztx_solved[unsafe_offset=q * k + j]
                )
            xpzx[unsafe_offset=i * k + j] = acc
            xpzx[unsafe_offset=j * k + i] = acc
    for i in range(k):
        var acc = 0.0
        for q in range(instruments):
            acc += ztx[unsafe_offset=q * k + i] * zty[unsafe_offset=q]
        p(beta)[unsafe_offset=i] = acc
    if not cholesky(xpzx, k):
        return 0
    cholesky_solve(xpzx, p(beta), k)
    return 1
