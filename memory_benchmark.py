"""
memory_benchmark.py
===================

Measure peak Python-heap and resident memory of each algorithm.

Two complementary measurements:

  - tracemalloc peak: the high-water mark of *Python-allocated* memory
    (NumPy ndarray buffers count toward this). Ignores memory held by
    the interpreter itself, JIT caches, or by C extensions outside the
    Python allocator.

  - RSS delta:        the change in resident set size around the call,
    via resource.getrusage(RUSAGE_SELF).ru_maxrss. This includes
    everything the OS thinks the process is holding.

Usage
-----
    python memory_benchmark.py --input inputs/input_1024.txt
    python memory_benchmark.py --input inputs/input_1024.txt --base-case 64

The script runs all four methods on the same input and prints a small
table comparing their memory footprints. Use this to show "Strassen now
needs ~one matrix's worth of scratch" instead of a much larger pile.
"""

from __future__ import annotations
import argparse
import gc
import os
import resource
import sys
import time
import tracemalloc

import numpy as np

from matrix_io import read_input
from algorithms import (
    Strassen,
    ParStrassen,
)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:7.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def measure(label: str, fn, *args, **kwargs):
    """Run fn(*args, **kwargs); return (elapsed, tracemalloc_peak, rss_delta)."""
    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Drop the result before measuring "after" RSS so we measure scratch.
    del result
    gc.collect()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    rss_high_water = max(rss_before, rss_after)
    return label, elapsed, peak, rss_high_water - rss_before


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Memory benchmark for the four methods.")
    p.add_argument("--input", required=True)
    p.add_argument("--base-case", type=int, default=64)
    p.add_argument("--cores", type=int, default=None)
    p.add_argument("--parallel-depth", type=int, default=1)
    args = p.parse_args(argv)

    print(f"[info] reading {args.input}")
    n, A, B = read_input(args.input)
    matrix_bytes = A.nbytes
    print(f"[info] matrix size n = {n}  (one int64 matrix = {_human_bytes(matrix_bytes)})")

    print("\nRunning each method and measuring scratch memory...\n")
    rows = []
    rows.append(measure("Strassen",     Strassen,     A, B, base_case=args.base_case))
    rows.append(measure("ParStrassen",  ParStrassen,  A, B,
                        base_case=args.base_case,
                        parallel_depth=args.parallel_depth,
                        cores=args.cores))

    # ----- Pretty print -----
    print(f"  {'Method':<14} {'Time (s)':>10} {'Py-heap peak':>14} "
          f"{'  ÷ matrix':>10}  {'RSS Δ':>14}")
    print("  " + "-" * 70)
    for label, t, peak, rss in rows:
        ratio = peak / matrix_bytes
        print(f"  {label:<14} {t:>10.4f} {_human_bytes(peak):>14} "
              f"{ratio:>9.2f}x {_human_bytes(rss):>14}")

    print()
    print("Notes:")
    print(f"  - 'Py-heap peak' is the additional Python/NumPy heap above the")
    print(f"    baseline at start of each call (tracemalloc).")
    print(f"  - '÷ matrix' is peak / size of one int64 n×n matrix.")
    print(f"    Strassen (optimized)     ≈ 2  (C plus geometric scratch).")
    print(f"    ParStrassen depth=1      higher: 7 separate M_i buffers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())