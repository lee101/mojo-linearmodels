"""Dense panel-data and IV kernels exposed through one C ABI compilation unit."""

from std.algorithm import sync_parallelize
from std.math import sqrt
from std.sys.info import simd_width_of

comptime W = simd_width_of[DType.float64]()
comptime PARALLEL_DIFFERENCE_THRESHOLD = 65536
comptime DIFFERENCE_GRAIN = 16384
comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def p(addr: Int) -> Ptr:
    return Ptr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def dot(a: Ptr, b: Ptr, n: Int) -> Float64:
    var acc = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        acc += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var total = acc.reduce_add()
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


def axpy(alpha: Float64, x: Ptr, y: Ptr, n: Int):
    var va = SIMD[DType.float64, W](alpha)
    var i = 0
    while i + W <= n:
        y.store(i, y.load[width=W](i) + va * x.load[width=W](i))
        i += W
    while i < n:
        y[i] += alpha * x[i]
        i += 1


def gram(x: Ptr, dst: Ptr, n: Int, d: Int):
    for i in range(d * d):
        dst[i] = 0.0
    for r in range(n):
        var row = x + r * d
        for i in range(d):
            var value = row[i]
            if value != 0.0:
                axpy(value, row, dst + i * d, i + 1)
    for i in range(d):
        for j in range(i + 1, d):
            dst[i * d + j] = dst[j * d + i]


def cross(a: Ptr, b: Ptr, dst: Ptr, n: Int, da: Int, db: Int):
    for i in range(da * db):
        dst[i] = 0.0
    for r in range(n):
        var ar = a + r * da
        var br = b + r * db
        for i in range(da):
            var value = ar[i]
            if value != 0.0:
                axpy(value, br, dst + i * db, db)


def cholesky(a: Ptr, d: Int) -> Bool:
    for i in range(d):
        for j in range(i + 1):
            var acc = a[i * d + j]
            for k in range(j):
                acc -= a[i * d + k] * a[j * d + k]
            if i == j:
                if acc <= 0.0:
                    return False
                a[i * d + i] = sqrt(acc)
            else:
                a[i * d + j] = acc / a[j * d + j]
    return True


def cholesky_solve(l: Ptr, b: Ptr, d: Int):
    for i in range(d):
        var acc = b[i]
        for k in range(i):
            acc -= l[i * d + k] * b[k]
        b[i] = acc / l[i * d + i]
    for ri in range(d):
        var i = d - 1 - ri
        var acc = b[i]
        for k in range(i + 1, d):
            acc -= l[k * d + i] * b[k]
        b[i] = acc / l[i * d + i]


def solve_columns(l: Ptr, b: Ptr, rows: Int, cols: Int, work: Ptr):
    for j in range(cols):
        for i in range(rows):
            work[i] = b[i * cols + j]
        cholesky_solve(l, work, rows)
        for i in range(rows):
            b[i * cols + j] = work[i]


def ols(x: Ptr, y: Ptr, beta: Ptr, work: Ptr, n: Int, k: Int) -> Bool:
    gram(x, work, n, k)
    for j in range(k):
        var acc = 0.0
        for i in range(n):
            acc += x[i * k + j] * y[i]
        beta[j] = acc
    if not cholesky(work, k):
        return False
    cholesky_solve(work, beta, k)
    return True


def group_center_once(
    data: Ptr, weights: Ptr, codes: IPtr, n: Int, d: Int, groups: Int,
    means: Ptr, sums: Ptr
) -> Float64:
    for g in range(groups):
        sums[g] = 0.0
    for i in range(groups * d):
        means[i] = 0.0
    for i in range(n):
        var g = Int(codes[i])
        var wi = weights[i]
        sums[g] += wi
        axpy(wi, data + i * d, means + g * d, d)
    var largest = 0.0
    for g in range(groups):
        if sums[g] > 0.0:
            var inv = 1.0 / sums[g]
            for j in range(d):
                var value = means[g * d + j] * inv
                means[g * d + j] = value
                largest = max(largest, abs(value))
    for i in range(n):
        axpy(-1.0, means + Int(codes[i]) * d, data + i * d, d)
    return largest


@export("mlm_group_demean")
def mlm_group_demean(
    src: Int, weights: Int, codes: Int, dst: Int, means: Int, sums: Int,
    n: Int, d: Int, groups: Int
) abi("C"):
    var x = p(src)
    var w = p(weights)
    var result = p(dst)
    for i in range(n * d):
        result[i] = x[i]
    _ = group_center_once(result, w, ip(codes), n, d, groups, p(means), p(sums))
    for i in range(n):
        var root_w = sqrt(w[i])
        for j in range(d):
            result[i * d + j] *= root_w


@export("mlm_two_way_demean")
def mlm_two_way_demean(
    src: Int, weights: Int, entity: Int, time: Int, dst: Int,
    means: Int, sums: Int, n: Int, d: Int, entities: Int, periods: Int,
    max_iter: Int, tol: Float64
) abi("C") -> Int:
    var x = p(src)
    var w = p(weights)
    var result = p(dst)
    for i in range(n * d):
        result[i] = x[i]
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
    for i in range(n):
        var root_w = sqrt(w[i])
        for j in range(d):
            result[i * d + j] *= root_w
    return iterations


@export("mlm_group_mean")
def mlm_group_mean(
    src: Int, weights: Int, codes: Int, dst: Int, sums: Int,
    n: Int, d: Int, groups: Int
) abi("C"):
    var x = p(src)
    var w = p(weights)
    var result = p(dst)
    for i in range(groups * d):
        result[i] = 0.0
    for g in range(groups):
        p(sums)[g] = 0.0
    for i in range(n):
        var g = Int(ip(codes)[i])
        var wi = w[i]
        p(sums)[g] += wi
        axpy(wi, x + i * d, result + g * d, d)
    for g in range(groups):
        if p(sums)[g] > 0.0:
            var inv = 1.0 / p(sums)[g]
            for j in range(d):
                result[g * d + j] *= inv


def difference_row(x: Ptr, right: IPtr, result: Ptr, row: Int, d: Int):
    var source_row = Int(right[row])
    var current = x + source_row * d
    var previous = current - d
    var target = result + row * d
    var j = 0
    while j + W <= d:
        target.store(
            j,
            current.load[width=W](j) - previous.load[width=W](j),
        )
        j += W
    while j < d:
        target[j] = current[j] - previous[j]
        j += 1


def first_difference_serial(
    x: Ptr, right: IPtr, result: Ptr, rows: Int, d: Int
):
    for row in range(rows):
        difference_row(x, right, result, row, d)


def first_difference_parallel(
    src: Int, right_addr: Int, dst: Int, rows: Int, d: Int
):
    if rows < PARALLEL_DIFFERENCE_THRESHOLD:
        first_difference_serial(p(src), ip(right_addr), p(dst), rows, d)
        return

    var chunks = (rows + DIFFERENCE_GRAIN - 1) // DIFFERENCE_GRAIN

    @parameter
    def process_chunk(chunk: Int):
        var x = p(src)
        var right = ip(right_addr)
        var result = p(dst)
        var start = chunk * DIFFERENCE_GRAIN
        var end = min(start + DIFFERENCE_GRAIN, rows)
        for row in range(start, end):
            difference_row(x, right, result, row, d)

    sync_parallelize[process_chunk](chunks)


@export("mlm_first_difference")
def mlm_first_difference(
    src: Int, right: Int, dst: Int, rows: Int, d: Int
) abi("C"):
    first_difference_parallel(src, right, dst, rows, d)


@export("mlm_ols")
def mlm_ols(
    x: Int, y: Int, beta: Int, work: Int, n: Int, k: Int
) abi("C") -> Int:
    return 1 if ols(p(x), p(y), p(beta), p(work), n, k) else 0


@export("mlm_cross")
def mlm_cross(
    a: Int, b: Int, dst: Int, n: Int, da: Int, db: Int
) abi("C"):
    cross(p(a), p(b), p(dst), n, da, db)


@export("mlm_predict")
def mlm_predict(
    x: Int, beta: Int, dst: Int, n: Int, k: Int
) abi("C"):
    for i in range(n):
        p(dst)[i] = dot(p(x) + i * k, p(beta), k)


@export("mlm_iv2sls")
def mlm_iv2sls(
    x: Int, y: Int, z: Int, beta: Int, work: Int,
    n: Int, k: Int, instruments: Int
) abi("C") -> Int:
    var xp = p(x)
    var yp = p(y)
    var zp = p(z)
    var wp = p(work)
    var ztz = wp
    var ztx = ztz + instruments * instruments
    var ztx_solved = ztx + instruments * k
    var zty = ztx_solved + instruments * k
    var xpzx = zty + instruments
    var temp = xpzx + k * k

    gram(zp, ztz, n, instruments)
    if not cholesky(ztz, instruments):
        return 0
    cross(zp, xp, ztx, n, instruments, k)
    for j in range(instruments):
        var acc = 0.0
        for i in range(n):
            acc += zp[i * instruments + j] * yp[i]
        zty[j] = acc
    for i in range(instruments * k):
        ztx_solved[i] = ztx[i]
    solve_columns(ztz, ztx_solved, instruments, k, temp)
    cholesky_solve(ztz, zty, instruments)

    for i in range(k):
        for j in range(i + 1):
            var acc = 0.0
            for q in range(instruments):
                acc += ztx[q * k + i] * ztx_solved[q * k + j]
            xpzx[i * k + j] = acc
            xpzx[j * k + i] = acc
    for i in range(k):
        var acc = 0.0
        for q in range(instruments):
            acc += ztx[q * k + i] * zty[q]
        p(beta)[i] = acc
    if not cholesky(xpzx, k):
        return 0
    cholesky_solve(xpzx, p(beta), k)
    return 1
