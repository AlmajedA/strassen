"""
generate_input.py
=================

Produce a random input file that matches the project's input format:

    First line: n
    Then 2 * n * n integer values in [-9, 9], row-major: A then B,
    written 32 per line for compactness.

Usage
-----
    python generate_input.py -n 256 -o inputs/input_256.txt
    python generate_input.py -n 1024 -o inputs/input_1024.txt --seed 1
"""

from __future__ import annotations
import argparse
import os
import sys
import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate Strassen project test inputs.")
    p.add_argument("-n", "--size", type=int, required=True,
                   help="Matrix dimension (must be a power of 2).")
    p.add_argument("-o", "--output", required=True,
                   help="Output file path.")
    p.add_argument("--seed", type=int, default=12,
                   help="RNG seed for reproducibility.")
    p.add_argument("--low", type=int, default=-9, help="Min value (inclusive).")
    p.add_argument("--high", type=int, default=9, help="Max value (inclusive).")
    args = p.parse_args(argv)

    n = args.size
    if n <= 0 or (n & (n - 1)) != 0:
        print(f"error: n must be a positive power of 2 (got {n})", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rng = np.random.default_rng(args.seed)
    # high+1 because numpy.integers' upper bound is exclusive.
    A = rng.integers(args.low, args.high + 1, size=(n, n), dtype=np.int64)
    B = rng.integers(args.low, args.high + 1, size=(n, n), dtype=np.int64)
    flat = np.concatenate([A.ravel(), B.ravel()])

    per_line = 32
    with open(args.output, "w") as f:
        f.write(f"{n}\n")
        for i in range(0, flat.size, per_line):
            row = flat[i : i + per_line]
            f.write(" ".join(str(int(x)) for x in row))
            f.write("\n")

    print(f"Wrote {args.output} ({n}x{n}, seed={args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
