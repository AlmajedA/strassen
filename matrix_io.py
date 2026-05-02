"""
matrix_io.py
============

Input/output handling for the Strassen project.

Input format
------------
A single text file:
    - First whitespace token: integer n  (matrix dimension, must be a power of 2).
    - Next 2 * n * n integer tokens: the entries of A then B in row-major order.
    - Whitespace between tokens may be any combination of spaces, tabs, newlines.
    - Each entry is in the range [-9, 9] but we never depend on that.

We parse with int64 throughout because for n = 16384 with values in [-9, 9],
a worst-case dot product is bounded by 9 * 9 * 16384 ≈ 1.3e6, well within int64
(±9.2e18). Using int64 gives plenty of headroom and matches the project spec.

Output format
-------------
Result file:    "<base>-<n>-output-<method>.txt"
    - First line: n
    - Next n lines: n integers per row, space-separated

Info file:      "<base>-<n>-info-<method>.txt"
    - "key: value" lines reporting method, dimension, timing, cores, etc.
"""

from __future__ import annotations
import os
import numpy as np


# --------------------------------------------------------------------------- #
# Reading                                                                     #
# --------------------------------------------------------------------------- #

def read_input(path: str):
    """
    Parse an input file and return (n, A, B) where A, B are np.int64 arrays.

    Memory-conscious version
    ------------------------
    The previous implementation did `f.read().split()` which loads the
    *entire* text file into RAM as a Python string, then builds a list
    of millions of Python str objects, then converts. For n = 16384 the
    text file is roughly 1.5 GiB and the intermediate Python objects
    push that to several gigabytes of transient memory just to read.

    This version reads the first line for `n`, then hands the remaining
    file descriptor to NumPy's C-level whitespace-text parser
    (`np.fromfile(f, dtype=np.int64, sep=' ')`). It streams through the
    file without ever materialising the full text in Python objects,
    so peak parse-time memory is dominated by the (unavoidable) target
    arrays A and B themselves.
    """
    with open(path, "rb") as f:
        # First line gives n. readline() reads up to and including the
        # newline; a few bytes, no risk of blowing memory.
        first_line = f.readline()
        if not first_line:
            raise ValueError(f"Input file '{path}' is empty.")
        try:
            n = int(first_line.strip())
        except ValueError:
            raise ValueError(
                f"Input file '{path}' first line is not an integer: "
                f"{first_line[:80]!r}"
            )
        if n <= 0 or (n & (n - 1)) != 0:
            raise ValueError(f"n must be a positive power of 2; got {n}.")

        expected = 2 * n * n
        # NumPy's text fromfile is a C parser and does not buffer the
        # whole file into a Python string. Stops after `count` items.
        data = np.fromfile(f, dtype=np.int64, sep=" ", count=expected)

    if data.size < expected:
        raise ValueError(
            f"Input '{path}' contains {data.size} matrix entries but "
            f"expected 2*n*n = {expected} for n = {n}."
        )

    # `data` is contiguous, so reshape returns contiguous views into it.
    A = data[: n * n].reshape(n, n)
    B = data[n * n :].reshape(n, n)
    return n, A, B


# --------------------------------------------------------------------------- #
# Writing                                                                     #
# --------------------------------------------------------------------------- #

def write_result(path: str, C: np.ndarray) -> None:
    """Write the result matrix in the format: n on line 1, then n rows."""
    n = C.shape[0]
    # Use np.savetxt-style fast writing.
    with open(path, "w") as f:
        f.write(f"{n}\n")
        # np.savetxt formats numbers fast; we drive it via f for header control.
        np.savetxt(f, C, fmt="%d")


def write_info(path: str, info: dict) -> None:
    """Write key: value pairs, one per line, in insertion order."""
    with open(path, "w") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def format_hms(seconds: float) -> str:
    """Format a duration as hh:mm:ss.sss with millisecond precision."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - 3600 * h - 60 * m
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def output_paths(input_path: str, n: int, method: str, output_dir: str):
    """Return the (result_path, info_path) tuple per the spec."""
    base = os.path.splitext(os.path.basename(input_path))[0]
    result = os.path.join(output_dir, f"{base}-{n}-output-{method}.txt")
    info = os.path.join(output_dir, f"{base}-{n}-info-{method}.txt")
    return result, info


def verify_against_numpy(C: np.ndarray, A: np.ndarray, B: np.ndarray) -> bool:
    """
    Trusted-result correctness check.

    We compare against NumPy's matmul, which dispatches to the platform BLAS
    for floats but uses a tight C kernel for int64. For the small/moderate
    sizes used in tests this is plenty fast and is the standard 'oracle'.
    """
    expected = A @ B  # both already int64 -> int64 result
    return np.array_equal(C, expected)
