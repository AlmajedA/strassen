"""
main.py
=======

Command-line entry point.

Examples
--------
    python main.py --input input1.txt --method Strassen      --base-case 64 --output-dir results
    python main.py --input input1.txt --method ParStrassen   --cores 8 --base-case 64 \
                   --parallel-depth 2 --output-dir results

    Add --verify to also compare against NumPy's matmul (the trusted oracle).
"""

from __future__ import annotations
import argparse
import os
import sys
import time

from matrix_io import (
    read_input,
    write_result,
    write_info,
    output_paths,
    format_hms,
    verify_against_numpy,
)
from algorithms import (
    Strassen,
    ParStrassen,
)
from algorithms_omp import get_max_threads


METHODS = ("Strassen", "ParStrassen")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Sequential & parallel Strassen matrix multiplication."
    )
    p.add_argument("--input", required=True, help="Path to input file.")
    p.add_argument("--method", required=True, choices=METHODS,
                   help="Which algorithm to run.")
    p.add_argument("--cores", type=int, default=None,
                   help="Number of threads (ParStrassen only). "
                        "Defaults to all available logical cores.")
    p.add_argument("--base-case", type=int, default=64,
                   help="Strassen recursion base-case size (must be a power of 2).")
    p.add_argument("--parallel-depth", type=int, default=2,
                   help="ParStrassen: how many recursion levels spawn tasks.")
    p.add_argument("--output-dir", default="results",
                   help="Directory to write result and info files.")
    p.add_argument("--verify", action="store_true",
                   help="Cross-check the result against NumPy's matmul.")
    return p.parse_args(argv)


def run(args) -> int:
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[info] reading {args.input}")
    n, A, B = read_input(args.input)
    print(f"[info] matrix size n = {n}")

    if args.base_case <= 0 or (args.base_case & (args.base_case - 1)) != 0:
        print(f"[warn] base-case {args.base_case} is not a power of 2.",
              file=sys.stderr)
    if args.base_case > n:
        print(f"[warn] base-case {args.base_case} >= n {n}; "
              "Strassen will trivially fall back to standard mult.",
              file=sys.stderr)

    cores_for_info = (args.cores if args.cores else get_max_threads()) \
                     if args.method == "ParStrassen" else 1

    # --------------------------------------------------------------------- #
    # Timed region: ONLY the multiplication.                                 #
    # --------------------------------------------------------------------- #
    print(f"[info] running {args.method}...")
    t0 = time.perf_counter()
    if args.method == "Strassen":
        C = Strassen(A, B, base_case=args.base_case)
    elif args.method == "ParStrassen":
        C = ParStrassen(
            A, B,
            base_case=args.base_case,
            parallel_depth=args.parallel_depth,
            cores=args.cores,
        )
    else:
        raise ValueError(args.method)
    elapsed = time.perf_counter() - t0
    print(f"[info] {args.method} finished in {elapsed:.4f}s")

    # --------------------------------------------------------------------- #
    # Write outputs.                                                         #
    # --------------------------------------------------------------------- #
    result_path, info_path = output_paths(args.input, n, args.method, args.output_dir)
    write_result(result_path, C)

    info = {
        "Method":               args.method,
        "Matrix dimension":     n,
        "Time (hh-mm-ss)":      format_hms(elapsed),
        "Time (seconds)":       f"{elapsed:.6f}",
        "Cores used":           cores_for_info,
        "Strassen base case":   args.base_case,
    }
    if args.method == "ParStrassen":
        info["Parallel depth"] = args.parallel_depth

    if args.verify:
        ok = verify_against_numpy(C, A, B)
        info["Verification"] = "PASS" if ok else "FAIL"
        print(f"[verify] {'PASS' if ok else 'FAIL'}")

    write_info(info_path, info)
    print(f"[info] wrote {result_path}")
    print(f"[info] wrote {info_path}")
    return 0 if (not args.verify or info.get("Verification") == "PASS") else 2


if __name__ == "__main__":
    sys.exit(run(parse_args()))