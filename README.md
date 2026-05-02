# Strassen Matrix Multiplication — Sequential and Parallel

A complete senior-level project comparing four algorithms for `n × n` integer
matrix multiplication on shared-memory hardware:

| Method         | Algorithm                       | Parallelism            | Complexity              |
| -------------- | ------------------------------- | ---------------------- | ----------------------- |
| `Sequential`   | Standard triple loop            | none (single thread)   | O(n³)                   |
| `ParMtrixMult` | Standard triple loop            | data, row-parallel     | O(n³ / p) + overhead    |
| `Strassen`     | Strassen recursive (7 products) | none (single thread)   | O(n^log₂ 7) ≈ O(n^2.807)|
| `ParStrassen`  | Strassen recursive              | task, 7 sub-products   | O(n^log₂ 7 / min(p,7^d))|

The four method names match the project specification exactly.

> **Defense-quality detail:** the project report
> [`REPORT.md`](REPORT.md) covers algorithms, theory,
> shared-memory model, experimental setup, results, speedup/efficiency, and
> memory limitations.

---

## Files

```
.
├── main.py              # CLI entry point
├── algorithms.py        # The four algorithms (imports Cython kernels)
├── algorithms_omp.pyx   # Cython+OpenMP kernels (seq_kernel, par_kernel)
├── setup.py             # Build config for Cython with -fopenmp
├── matrix_io.py         # Input parsing (streaming), output writing, helpers
├── benchmark.py         # Aggregates info files into CSV + Markdown
├── generate_input.py    # Random input generator
├── memory_benchmark.py  # Per-method peak-memory measurement
├── memory_compare.py    # Side-by-side: optimized vs naive Strassen memory
├── run_experiments.sh   # Full experiment grid: build → input → run → summary
├── README.md            # this file
└── REPORT.md            # full academic report
```

---

## Memory optimizations applied (full detail in REPORT.md §11)

The Strassen and ParStrassen paths are written for large n:

1. **Destination-buffer passing** — recursive calls write into a caller-supplied
   `C` instead of returning a fresh ndarray (saves one (n/2)² allocation per level).
2. **Compute-and-immediately-accumulate** — sequential Strassen never holds all
   M1..M7 at once; each is computed into a single shared buffer and immediately
   folded into the C quadrants. Per-level scratch drops from ~21 buffers of
   (n/2)² to just 3 (`Tx`, `Ty`, `M`).
3. **In-place NumPy ops** (`np.add(..., out=...)`, `np.copyto`, etc.) — no
   transient arrays from `M1+M4-M5+M7`-style expressions.
4. **Lazy input-sum allocation** — ParStrassen's 7 worker tasks build their
   own input sums; never coexist at the parent level.
5. **Streaming text input** — `np.fromfile` instead of reading the whole file
   into a Python string (avoids ~3 GB of transient strings at n = 16 384).

**Measured peak memory** (matrix-size multiples, via `tracemalloc`):

| Method        | Optimized | Naive textbook |
| ------------- | --------- | -------------- |
| Sequential    | 1.0×      | 1.0×           |
| ParMtrixMult  | 1.0×      | 1.0×           |
| **Strassen**      | **2.0×**      | **3.0×**           |
| ParStrassen   | 7.0×      | 9–10× (and worse with depth) |

Run `python memory_compare.py --input inputs/input_1024.txt` to verify
on your own machine.

---

## Dependencies

* Python ≥ 3.10
* NumPy
* Cython (compiles the OpenMP kernels)
* A C compiler with OpenMP support (GCC, Clang, or MSVC)

```bash
pip install numpy cython
```

---

## Build (required before first run)

The triple-loop kernels live in `algorithms_omp.pyx` and must be compiled
to a shared library before you can run anything:

```bash
python setup.py build_ext --inplace
```

This produces `algorithms_omp.*.so` (Linux/macOS) or `algorithms_omp.*.pyd`
(Windows). The generated C code contains literal `#pragma omp parallel for`
directives — you can inspect `algorithms_omp.c` to verify.

---

## Quick start

Generate an input and run all four methods:

```bash
# 0) build the Cython+OpenMP kernels (required once)
python setup.py build_ext --inplace

# 1) generate an input matrix
python generate_input.py -n 256 -o inputs/input_256.txt

# 2) run each method (results land in ./results/)
python main.py --input inputs/input_256.txt --method Sequential   --output-dir results --verify
python main.py --input inputs/input_256.txt --method ParMtrixMult --cores 8 --output-dir results --verify
python main.py --input inputs/input_256.txt --method Strassen     --base-case 64 --output-dir results --verify
python main.py --input inputs/input_256.txt --method ParStrassen  --cores 8 --base-case 64 --parallel-depth 2 --output-dir results --verify

# 3) aggregate into a summary CSV + Markdown table
python benchmark.py results
```

Or run the entire experiment grid (sweeps n, cores, base-case, parallel-depth)
and produce a summary in one shot:

```bash
bash run_experiments.sh
```

---

## CLI reference

```
python main.py --input <FILE> --method <METHOD> [options]
```

| Flag                | Required | Default | Notes                                                   |
| ------------------- | :------: | :-----: | ------------------------------------------------------- |
| `--input`           | ✓        |   —     | Path to the input file (n + 2n² ints)                   |
| `--method`          | ✓        |   —     | `Sequential` / `ParMtrixMult` / `Strassen` / `ParStrassen` |
| `--cores`           |          | all CPUs| Threads for parallel methods                            |
| `--base-case`       |          | 64      | Strassen recursion threshold (power of 2)               |
| `--parallel-depth`  |          | 2       | ParStrassen: how many recursion levels spawn tasks      |
| `--output-dir`      |          | results | Directory for the result and info files                 |
| `--verify`          |          | off     | Cross-check against NumPy's matmul (the trusted oracle) |

### Input file format

```
<n>
<2 * n * n integers, whitespace-separated, any line widths>
```

The first `n²` integers are matrix `A`, the next `n²` are matrix `B`, both in
row-major order. We parse with `int64`.

### Output files

For input `input1.txt` and `n = 128`, four method runs produce:

```
input1-128-output-Sequential.txt        input1-128-info-Sequential.txt
input1-128-output-ParMtrixMult.txt      input1-128-info-ParMtrixMult.txt
input1-128-output-Strassen.txt          input1-128-info-Strassen.txt
input1-128-output-ParStrassen.txt       input1-128-info-ParStrassen.txt
```

Result files: line 1 has `n`, then `n` rows of `n` integers each.

Info files contain:

```
Method: ParStrassen
Matrix dimension: 128
Time (hh-mm-ss): 00:00:00.066
Time (seconds): 0.065663
Cores used: 2
Strassen base case: 32
Parallel depth: 2
Verification: PASS         # only when --verify was passed
```

---

## Implementation strategy (TL;DR — full reasoning in REPORT.md)

* **Why not `multiprocessing`?** It would need to pickle 2 GB matrices across
  process boundaries — that is the *distributed*-memory model, not shared.
* **Why not pure `threading`?** The CPython GIL serialises pure-Python
  bytecode; threads can't execute Python loops in parallel (unless you use
  Python 3.14t free-threaded builds — see REPORT.md §5).
* **What we use:**
  * **Cython + OpenMP** for `ParMtrixMult`. The triple-loop kernel in
    `algorithms_omp.pyx` uses `cython.parallel.prange` which compiles to
    a real `#pragma omp parallel for schedule(static) num_threads(nt)`.
    You can verify this in the generated `algorithms_omp.c`.
  * **`ThreadPoolExecutor`** for `ParStrassen` task parallelism. The Cython
    kernels release the GIL (`with nogil:`) while running, so the seven
    Strassen sub-products run on different cores in true parallel.

Thread count for `ParMtrixMult` is controlled by `--cores` (maps to
OpenMP's `num_threads` clause) or by the `OMP_NUM_THREADS` environment
variable.

---

## Running on a small machine

`run_experiments.sh` defaults to `n ∈ {64, 128, 256, 512, 1024}` because
anything larger is slow on the sequential triple-loop baseline. To extend:

```bash
SIZES="64 128 256 512 1024 2048" bash run_experiments.sh
```

For correctness verification on huge `n`, the NumPy oracle is also slow on
`int64`; the script auto-skips `--verify` for `n > VERIFY_MAX_N` (default
`1024`). Override via `VERIFY_MAX_N=4096 bash run_experiments.sh`.

A note on `n = 16 384`: with `int64`, each matrix is 2 GB, and Strassen
needs many temporaries — peak RSS comfortably exceeds 16 GB.
See REPORT.md §11 for the full memory analysis.
# strassen
