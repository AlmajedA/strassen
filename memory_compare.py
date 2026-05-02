"""
memory_compare.py
=================

Side-by-side comparison: the *optimized* Strassen vs a *naive textbook*
Strassen on the same input. The naive one returns a fresh ndarray from
every recursive call and holds all 7 M_i simultaneously, which is the
default a student would write before optimizing.

Run:
    python memory_compare.py --input inputs/input_512.txt --base-case 64

It prints the ratio of memory peaks so you can quote a concrete number.
"""

from __future__ import annotations
import argparse
import gc
import sys
import time
import tracemalloc

import numpy as np

from matrix_io import read_input
from algorithms import Strassen as Strassen_optimized
from algorithms_omp import seq_kernel


# ----------------------------------------------------------------------- #
# Reference textbook implementation (the "before" version).               #
# Returns a fresh ndarray, builds all input sums upfront, holds M1..M7.   #
# ----------------------------------------------------------------------- #
def Strassen_naive(A, B, base_case=64):
    n = A.shape[0]
    if n <= base_case:
        C = np.zeros((n, n), dtype=np.int64)
        seq_kernel(A, B, C)
        return C

    h = n // 2
    A11, A12 = A[:h, :h], A[:h, h:]
    A21, A22 = A[h:, :h], A[h:, h:]
    B11, B12 = B[:h, :h], B[:h, h:]
    B21, B22 = B[h:, :h], B[h:, h:]

    M1 = Strassen_naive(A11 + A22, B11 + B22, base_case)
    M2 = Strassen_naive(A21 + A22, B11,        base_case)
    M3 = Strassen_naive(A11,        B12 - B22, base_case)
    M4 = Strassen_naive(A22,        B21 - B11, base_case)
    M5 = Strassen_naive(A11 + A12, B22,        base_case)
    M6 = Strassen_naive(A21 - A11, B11 + B12, base_case)
    M7 = Strassen_naive(A12 - A22, B21 + B22, base_case)

    C = np.empty((n, n), dtype=np.int64)
    C[:h, :h] = M1 + M4 - M5 + M7
    C[:h, h:] = M3 + M5
    C[h:, :h] = M2 + M4
    C[h:, h:] = M1 - M2 + M3 + M6
    return C


def _human_bytes(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:7.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TiB"


def measure(label, fn, *args, **kwargs):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, elapsed, peak


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--base-case", type=int, default=64)
    args = p.parse_args(argv)

    n, A, B = read_input(args.input)
    matrix_bytes = A.nbytes
    print(f"n = {n};  one int64 matrix = {_human_bytes(matrix_bytes)}")

    print(f"\nbase-case = {args.base_case}\n")
    print(f"  {'Implementation':<28} {'Time (s)':>10} {'Py-heap peak':>14}  {'÷ matrix':>9}")
    print("  " + "-" * 70)

    C_naive,     t_naive,     peak_naive     = measure("Strassen (naive)",     Strassen_naive,     A, B, args.base_case)
    C_optimized, t_optimized, peak_optimized = measure("Strassen (optimized)", Strassen_optimized, A, B, args.base_case)

    print(f"  {'Strassen (naive)':<28} {t_naive:>10.4f} "
          f"{_human_bytes(peak_naive):>14}  {peak_naive/matrix_bytes:>8.2f}x")
    print(f"  {'Strassen (optimized)':<28} {t_optimized:>10.4f} "
          f"{_human_bytes(peak_optimized):>14}  {peak_optimized/matrix_bytes:>8.2f}x")

    print()
    print(f"Memory reduction: {peak_naive / peak_optimized:.2f}x "
          f"({_human_bytes(peak_naive - peak_optimized)} saved)")

    # Sanity: results must match.
    if np.array_equal(C_naive, C_optimized):
        print("Correctness:      both implementations match ✓")
    else:
        print("Correctness:      MISMATCH ✗")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
