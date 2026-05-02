"""
benchmark.py
============

Aggregates the info files produced by `main.py` into a single CSV and
Markdown summary, with speedup and efficiency computed against the
Sequential baseline.

Usage
-----
    python benchmark.py [results_dir]

If results_dir is omitted, "results" is assumed.

Speedup and efficiency
----------------------
    speedup_p   = T_sequential(n) / T_p(n)
    efficiency  = speedup_p / p

We compute these against the *same n* sequential baseline. If no
sequential run exists for a given n, we leave the columns blank.
"""

from __future__ import annotations
import csv
import glob
import os
import sys


# Parse a single info file into a dict.
def parse_info(path: str) -> dict:
    info = {"file": os.path.basename(path)}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    return info


def collect(results_dir: str, recursive: bool = True) -> list[dict]:
    pattern = "**/*-info-*.txt" if recursive else "*-info-*.txt"
    paths = sorted(glob.glob(os.path.join(results_dir, pattern), recursive=recursive))
    return [parse_info(p) for p in paths]


def add_speedup(rows: list[dict]) -> list[dict]:
    """Compute Speedup and Efficiency vs Strassen baseline at the same n.

    If multiple Strassen runs exist for the same n (e.g. different base cases),
    we use the *fastest* as the baseline.
    """
    baseline: dict[str, float] = {}
    for r in rows:
        if r.get("Method") == "Strassen":
            n = r.get("Matrix dimension", "")
            try:
                t = float(r.get("Time (seconds)", "nan"))
            except ValueError:
                continue
            if n and (n not in baseline or t < baseline[n]):
                baseline[n] = t

    for r in rows:
        n = r.get("Matrix dimension", "")
        try:
            t = float(r.get("Time (seconds)", "nan"))
        except ValueError:
            t = float("nan")
        if n in baseline and t > 0 and baseline[n] > 0:
            sp = baseline[n] / t
            r["Speedup vs Strassen"] = f"{sp:.3f}"
            try:
                cores = int(r.get("Cores used", "1"))
            except ValueError:
                cores = 1
            r["Efficiency"] = f"{sp / cores:.3f}" if cores > 0 else ""
        else:
            r["Speedup vs Strassen"] = ""
            r["Efficiency"] = ""
    return rows


# --------------------------------------------------------------------------- #
# Output writers                                                              #
# --------------------------------------------------------------------------- #

CSV_COLS = [
    "file",
    "Method",
    "Matrix dimension",
    "Time (hh-mm-ss)",
    "Time (seconds)",
    "Cores used",
    "Strassen base case",
    "Parallel depth",
    "Speedup vs Strassen",
    "Efficiency",
    "Verification",
]


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})


def write_markdown(rows: list[dict], path: str) -> None:
    cols = [c for c in CSV_COLS if c != "file"]
    with open(path, "w") as f:
        f.write("# Benchmark Summary\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    args = argv or sys.argv[1:]
    results_dir = args[0] if args else "results"
    if not os.path.isdir(results_dir):
        print(f"error: '{results_dir}' is not a directory", file=sys.stderr)
        return 1

    rows = collect(results_dir)
    if not rows:
        print(f"warning: no '*-info-*.txt' files found under {results_dir}",
              file=sys.stderr)
        return 0

    rows = add_speedup(rows)

    # Sort by (n, method) for a tidy table.
    def key(r):
        try:
            n = int(r.get("Matrix dimension", "0"))
        except ValueError:
            n = 0
        return (n, r.get("Method", ""), r.get("Cores used", ""), r.get("file", ""))
    rows.sort(key=key)

    csv_path = os.path.join(results_dir, "summary.csv")
    md_path = os.path.join(results_dir, "summary.md")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Aggregated {len(rows)} runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())