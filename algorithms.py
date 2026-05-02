"""
algorithms.py
=============

Two Strassen algorithms:

    Strassen        - Recursive divide-and-conquer, 7 sub-products, sequential.
    ParStrassen     - Recursive Strassen with the 7 sub-products dispatched
                      to a thread pool down to a configurable depth.

Both use a Cython-compiled ``seq_kernel`` (``algorithms_omp.pyx``) as the
base-case multiplier. That kernel releases the GIL (``with nogil:``),
allowing the ParStrassen thread pool to run tasks on separate cores.

Memory optimizations
--------------------
  (1) DESTINATION-BUFFER PASSING — recursive calls write into a
      caller-provided C, not a fresh ndarray.
  (2) COMPUTE-AND-ACCUMULATE — only 3 scratch buffers per recursion level.
  (3) IN-PLACE NUMPY OPS — ``np.add(..., out=...)``, etc.
  (4) LAZY INPUT-SUM ALLOCATION in the parallel path.

Per-level scratch: 3 * (n/2)^2 buffers.
Total scratch across all levels: n^2 int64 = 8 n^2 bytes.
"""

from __future__ import annotations
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from algorithms_omp import seq_kernel, get_max_threads


# --------------------------------------------------------------------------- #
# 1. Sequential Strassen (in-place, compute-and-accumulate)                   #
# --------------------------------------------------------------------------- #
#
# Strassen's identity: 7 sub-products instead of 8.
#   M1 = (A11+A22)(B11+B22)        C11 = M1 + M4 - M5 + M7
#   M2 = (A21+A22) B11             C12 = M3 + M5
#   M3 = A11 (B12-B22)             C21 = M2 + M4
#   M4 = A22 (B21-B11)             C22 = M1 - M2 + M3 + M6
#   M5 = (A11+A12) B22
#   M6 = (A21-A11)(B11+B12)
#   M7 = (A12-A22)(B21+B22)
#
# Recurrence:  T(n) = 7 T(n/2) + Θ(n^2)  =>  Θ(n^{log2 7}) ≈ n^2.807
#
# Each level allocates 3 scratch buffers (Tx, Ty, M). M is reused for
# all 7 products in turn; each M_i is immediately folded into C.

def _strassen_into(A: np.ndarray,
                   B: np.ndarray,
                   C: np.ndarray,
                   base_case: int) -> None:
    """Compute A @ B in place into C."""
    n = A.shape[0]
    if n <= base_case:
        seq_kernel(A, B, C)
        return

    h = n // 2
    A11, A12 = A[:h, :h], A[:h, h:]
    A21, A22 = A[h:, :h], A[h:, h:]
    B11, B12 = B[:h, :h], B[:h, h:]
    B21, B22 = B[h:, :h], B[h:, h:]
    C11, C12 = C[:h, :h], C[:h, h:]
    C21, C22 = C[h:, :h], C[h:, h:]

    Tx = np.empty((h, h), dtype=np.int64)
    Ty = np.empty((h, h), dtype=np.int64)
    M  = np.empty((h, h), dtype=np.int64)

    # M1 = (A11 + A22)(B11 + B22)
    np.add(A11, A22, out=Tx)
    np.add(B11, B22, out=Ty)
    _strassen_into(Tx, Ty, M, base_case)
    np.copyto(C11, M)
    np.copyto(C22, M)

    # M2 = (A21 + A22) * B11
    np.add(A21, A22, out=Tx)
    _strassen_into(Tx, B11, M, base_case)
    np.copyto(C21, M)
    np.subtract(C22, M, out=C22)

    # M3 = A11 * (B12 - B22)
    np.subtract(B12, B22, out=Ty)
    _strassen_into(A11, Ty, M, base_case)
    np.copyto(C12, M)
    np.add(C22, M, out=C22)

    # M4 = A22 * (B21 - B11)
    np.subtract(B21, B11, out=Ty)
    _strassen_into(A22, Ty, M, base_case)
    np.add(C11, M, out=C11)
    np.add(C21, M, out=C21)

    # M5 = (A11 + A12) * B22
    np.add(A11, A12, out=Tx)
    _strassen_into(Tx, B22, M, base_case)
    np.subtract(C11, M, out=C11)
    np.add(C12, M, out=C12)

    # M6 = (A21 - A11)(B11 + B12)
    np.subtract(A21, A11, out=Tx)
    np.add(B11, B12, out=Ty)
    _strassen_into(Tx, Ty, M, base_case)
    np.add(C22, M, out=C22)

    # M7 = (A12 - A22)(B21 + B22)
    np.subtract(A12, A22, out=Tx)
    np.add(B21, B22, out=Ty)
    _strassen_into(Tx, Ty, M, base_case)
    np.add(C11, M, out=C11)


def Strassen(A: np.ndarray, B: np.ndarray, base_case: int = 64) -> np.ndarray:
    """Sequential recursive Strassen with configurable base-case size."""
    n = A.shape[0]
    C = np.empty((n, n), dtype=np.int64)
    _strassen_into(A, B, C, base_case)
    return C


# --------------------------------------------------------------------------- #
# 2. Parallel Strassen (task-parallel, in-place)                              #
# --------------------------------------------------------------------------- #
#
# The 7 sub-products M1..M7 are independent and dispatched to a
# ThreadPoolExecutor.  The Cython base-case kernel releases the GIL
# (``with nogil:``), so threads truly run on different cores.
#
# parallel_depth bounds the task tree: below that depth we fall through
# to the allocation-light sequential ``_strassen_into``.

def _par_strassen_into(A: np.ndarray,
                       B: np.ndarray,
                       C: np.ndarray,
                       base_case: int,
                       depth: int) -> None:
    """Parallel in-place Strassen. Writes A @ B into C."""
    n = A.shape[0]
    if n <= base_case:
        seq_kernel(A, B, C)
        return
    if depth <= 0:
        _strassen_into(A, B, C, base_case)
        return

    h = n // 2
    A11, A12 = A[:h, :h], A[:h, h:]
    A21, A22 = A[h:, :h], A[h:, h:]
    B11, B12 = B[:h, :h], B[:h, h:]
    B21, B22 = B[h:, :h], B[h:, h:]
    C11, C12 = C[:h, :h], C[:h, h:]
    C21, C22 = C[h:, :h], C[h:, h:]

    M = [np.empty((h, h), dtype=np.int64) for _ in range(7)]

    def _run(idx, lhs_fn, rhs_fn):
        L = lhs_fn()
        R = rhs_fn()
        _par_strassen_into(L, R, M[idx], base_case, depth - 1)

    tasks = [
        (0, lambda: A11 + A22, lambda: B11 + B22),
        (1, lambda: A21 + A22, lambda: B11        ),
        (2, lambda: A11,        lambda: B12 - B22),
        (3, lambda: A22,        lambda: B21 - B11),
        (4, lambda: A11 + A12, lambda: B22        ),
        (5, lambda: A21 - A11, lambda: B11 + B12),
        (6, lambda: A12 - A22, lambda: B21 + B22),
    ]

    with ThreadPoolExecutor(max_workers=7) as ex:
        futures = [ex.submit(_run, *t) for t in tasks]
        for f in futures:
            f.result()

    M1, M2, M3, M4, M5, M6, M7 = M

    np.add(M1, M4, out=C11)
    np.subtract(C11, M5, out=C11)
    np.add(C11, M7, out=C11)

    np.add(M3, M5, out=C12)
    np.add(M2, M4, out=C21)

    np.subtract(M1, M2, out=C22)
    np.add(C22, M3, out=C22)
    np.add(C22, M6, out=C22)


def ParStrassen(A: np.ndarray,
                B: np.ndarray,
                base_case: int = 64,
                parallel_depth: int = 2,
                cores: int | None = None) -> np.ndarray:
    """Parallel Strassen with bounded task-creation depth."""
    n = A.shape[0]
    C = np.empty((n, n), dtype=np.int64)
    _par_strassen_into(A, B, C, base_case, parallel_depth)
    return C