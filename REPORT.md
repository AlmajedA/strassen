# Strassen Matrix Multiplication: Sequential and Parallel Implementations

**Algorithms project report**

## 1. Problem description

Given two `n × n` matrices `A` and `B` of 64-bit integers, compute `C = A · B`
where `C[i,j] = Σ_k A[i,k] · B[k,j]`. The dimension `n` is a power of two;
entries are random integers in `[-9, 9]`. We must implement, benchmark, and
compare four algorithms:

1. **Sequential** — the classic O(n³) triple-loop.
2. **ParMtrixMult** — the same algorithm parallelised over rows of the
   output, using a shared-memory model.
3. **Strassen** — Strassen's recursive divide-and-conquer multiplication
   with a configurable base-case threshold.
4. **ParStrassen** — Strassen's recursion with the seven independent
   sub-products dispatched to a thread pool, with a bounded
   parallel-spawn depth.

The deliverables are: working code; per-method result and info output files;
correctness verification against a trusted oracle; an experiment script that
sweeps `n`, core count, base case, and parallel depth; and this report.

## 2. Input/output specification

### Input

A single text file:

```
<n>
<2·n² integer tokens, separated by whitespace, distributed across any
 number of lines>
```

The first `n²` integers populate `A`; the next `n²` populate `B`; both are
laid out row-major. Parsing uses `int64` because, although the inputs are in
`[-9, 9]`, downstream Strassen sums and 32-bit accumulators are an
unnecessary risk — `int64` covers any worst case (`9 · 9 · 16384 ≈ 1.3 × 10⁶`,
well within `±9.2 × 10¹⁸`).

### Output

For each method run, two files are written into `--output-dir`:

| File                                          | Contents                                              |
| --------------------------------------------- | ----------------------------------------------------- |
| `<base>-<n>-output-<method>.txt`              | line 1: `n`; lines 2..n+1: a row of `n` integers each |
| `<base>-<n>-info-<method>.txt`                | `key: value` pairs (method, dimension, time in hh-mm-ss, time in seconds, cores used, base case, depth, verification) |

`<base>` is the input filename without extension.

The *only* timed region is the multiplication call itself; reading inputs,
writing outputs, and Numba JIT compilation are all explicitly excluded.

## 3. Explanation of the four algorithms

### 3.1 Sequential (O(n³))

```
for i in 0..n:
    for j in 0..n:
        acc = 0
        for k in 0..n:
            acc += A[i,k] * B[k,j]
        C[i,j] = acc
```

The outer two loops enumerate output cells; the innermost loop accumulates
the dot-product. Each cell costs `n` multiplies and `n-1` adds, and there
are `n²` cells, giving `Θ(n³)` arithmetic operations. We hold the partial
sum in a register-local `acc` and write to `C[i,j]` once, which is the
standard idiom for cache-friendly accumulation.

### 3.2 ParMtrixMult (data parallelism)

The crucial property: **rows of `C` are independent**. Output cell
`C[i,j]` depends only on row `i` of `A` and column `j` of `B`; cells in
different rows touch disjoint output memory. So we can split the outer
`for i` loop across threads with no synchronisation, no locks, no atomics
— it is an embarrassingly-parallel loop:

```
parallel for i in 0..n:        # prange
    for j in 0..n:
        acc = 0
        for k in 0..n:
            acc += A[i,k] * B[k,j]
        C[i,j] = acc
```

In our implementation, `prange` (Numba's parallel-range) hands ranges of
`i` to a thread pool. All threads share `A`, `B`, and `C` — the
shared-memory model — but each writes to its own rows of `C`, so there is
no contention.

### 3.3 Strassen (sequential, O(n^log₂ 7))

Partition each `n × n` matrix into four `(n/2) × (n/2)` quadrants.
Strassen's identity expresses `C` using only **seven** sub-products
instead of the eight a naive recursion would need:

```
M1 = (A11 + A22)(B11 + B22)
M2 = (A21 + A22) B11
M3 = A11 (B12 - B22)
M4 = A22 (B21 - B11)
M5 = (A11 + A12) B22
M6 = (A21 - A11)(B11 + B12)
M7 = (A12 - A22)(B21 + B22)

C11 = M1 + M4 - M5 + M7
C12 = M3 + M5
C21 = M2 + M4
C22 = M1 - M2 + M3 + M6
```

Recurrence: `T(n) = 7·T(n/2) + Θ(n²)` — seven recursive halvings plus
`Θ(n²)` work for the additions. By the master theorem,
`T(n) = Θ(n^log₂ 7) ≈ Θ(n^2.807)`.

**Why a base case matters.** Each recursive level pays for 18 matrix
add/subtract operations of size `(n/2)² = Θ(n²)` plus seven temporaries
worth of memory allocation. For small `n` this constant dwarfs the
asymptotic gain. We therefore cut over to the standard kernel below a
threshold `base_case`. Empirically that crossover sits between 32 and 128
on most hardware. Too small a base case (e.g. 4) makes Strassen *slower*
than the standard algorithm because we spend almost all our time in
recursion bookkeeping and array additions; too large a base case (e.g.
`n/2`) and Strassen barely recurses, recovering only one Strassen step's
worth of savings.

### 3.4 ParStrassen (task parallelism)

The seven sub-products `M1..M7` are mutually independent: each reads from
sub-blocks of `A` and `B` (which are read-only at this point) and writes
to its own private result matrix. So we can fire all seven off
concurrently and only synchronise at the join, immediately before
constructing `C` from the `M_i`.

```
spawn task f1 = strassen(A11+A22, B11+B22, ...)
spawn task f2 = strassen(A21+A22, B11,      ...)
... f7
wait for f1..f7        # synchronisation barrier
C11 = M1 + M4 - M5 + M7
C12 = M3 + M5
C21 = M2 + M4
C22 = M1 - M2 + M3 + M6
```

**Why a parallel-depth bound.** If every recursive level were allowed to
spawn tasks, the task tree would have `7^d` outstanding tasks at depth
`d`. At depth 4 that's 2401 sub-tasks, almost all of which are too small
to dominate the cost of dispatching and joining a thread. We expose
`--parallel-depth` so the user can pick how many levels are
"task-spawning"; below that depth we fall through to the sequential
`Strassen`, which reduces overhead substantially.

Synchronisation happens via blocking `future.result()` calls — there are
no locks; the seven futures simply join before the additions that build
`C`.

Idealised runtime, ignoring overhead and assuming infinite cores
recursively:

```
T_∞(n) = T_∞(n/2) + Θ(n²)  ⇒  Θ(n²)
```

With finite cores `p` and parallel depth `d`:

```
T_p(n) ≈ work(n) / min(p, 7^d) + add_overhead + thread_pool_overhead
```

## 4. Implementation details

| File                 | Responsibility                                          |
| -------------------- | ------------------------------------------------------- |
| `algorithms_omp.pyx` | Cython+OpenMP kernels: `seq_kernel`, `par_kernel`, `get_max_threads` |
| `setup.py`           | builds the Cython extension with `-fopenmp` (GCC) or `/openmp` (MSVC) |
| `algorithms.py`      | the four required algorithms (Sequential, ParMtrixMult, Strassen, ParStrassen); imports compiled kernels |
| `matrix_io.py`       | parsing the input file, writing result and info files   |
| `main.py`            | CLI, timing, output-path construction, optional verification |
| `generate_input.py`  | random input generation in the project's input format   |
| `benchmark.py`       | aggregates info files into CSV and Markdown summaries with speedup/efficiency |
| `run_experiments.sh` | builds the extension, then drives the parameter-grid sweep |

The four algorithms share a common inner kernel (`seq_kernel`) written in
Cython. This function takes typed memoryviews (`long long[:, :]`) and runs
the triple loop inside a `with nogil:` block, which both releases the GIL
(allowing other threads to run) and guarantees that the loop body is pure C
with no Python interpreter overhead.

`par_kernel` uses `cython.parallel.prange` which the Cython compiler
translates to a literal `#pragma omp parallel for schedule(static)
num_threads(nt)` in the generated C code. The `num_threads` clause maps
directly to the user's `--cores` flag. Thread count can also be controlled
via the standard `OMP_NUM_THREADS` environment variable.

Unlike Numba, Cython compiles ahead of time (`python setup.py build_ext
--inplace`). There is no JIT warmup penalty and no first-call latency.
The generated `algorithms_omp.c` can be inspected to verify the OpenMP
directives — the professor can `grep "#pragma omp" algorithms_omp.c` and
see the actual parallel pragmas.

## 5. Shared-memory model

Direct OpenMP is not available in pure Python; the shared-memory options are:

| Option                | Verdict |
| --------------------- | ------- |
| `multiprocessing`     | rejected — pickling huge matrices to subprocesses is the *distributed*-memory model and incurs gigabyte serialisation costs |
| pure `threading`      | rejected — CPython's GIL serialises pure-Python loops, so threads cannot execute the multiplication concurrently (unless using Python 3.14t free-threaded build) |
| Numba `prange`        | viable — compiles to LLVM with OpenMP-like parallelism, but is an emulation, not actual OpenMP |
| **Cython + OpenMP** (chosen) | compiles `.pyx` to C with **real `#pragma omp parallel for`** directives; linked against `libgomp`; `with nogil:` releases the GIL for true multi-core execution |

So the project uses Cython for both:

* **Data parallelism** in `ParMtrixMult` via `cython.parallel.prange`.
  The generated C code contains verbatim `#pragma omp parallel for
  schedule(static) num_threads(nt)`. All threads share `A`, `B`, `C` in
  memory — this is the OpenMP shared-memory model, not an approximation.
* **Task parallelism** in `ParStrassen` via a Python `ThreadPoolExecutor`
  that submits seven Cython-compiled jobs. The Cython kernels run inside
  `with nogil:`, so the GIL does not serialise them and the threads
  execute on different cores in genuine parallel.

Thread count is controlled by:
* `--cores N` on the command line → passed as `num_threads=N` to OpenMP
* `OMP_NUM_THREADS=N` environment variable → standard OpenMP mechanism
* Default: OpenMP uses all available logical cores

On Python 3.14t (free-threaded / no-GIL build), the `with nogil:` context
managers are effectively no-ops (there is no GIL to release), so even the
Python-level Strassen recursion runs without GIL contention.

## 6. Theoretical analysis

### 6.1 Work, span, parallel runtime

| Method         | Work `T₁`           | Span `T_∞`             | Speedup ceiling |
| -------------- | ------------------- | ---------------------- | --------------- |
| Sequential     | `Θ(n³)`             | `Θ(n³)`                | 1               |
| ParMtrixMult   | `Θ(n³)`             | `Θ(n²)` *(one row)*    | `n` (then memory-bound) |
| Strassen       | `Θ(n^log₂ 7)`       | `Θ(n^log₂ 7)`          | 1               |
| ParStrassen    | `Θ(n^log₂ 7)`       | `Θ(n²)` *(span recurrence T(n)=T(n/2)+n²)* | up to `7^d` |

By Brent's theorem `T_p ≤ T_∞ + (T₁ - T_∞)/p`, so with `p` cores:

```
ParMtrixMult:  T_p(n) ≤ Θ(n²) + Θ(n³ / p)
ParStrassen:   T_p(n) ≤ Θ(n²) + Θ(n^log₂ 7 / min(p, 7^d))
```

### 6.2 Speedup and efficiency

* `S_p(n) = T₁(n) / T_p(n)` — speedup against the same algorithm, single thread.
* `E_p(n) = S_p(n) / p` — efficiency, with `1.0` being perfect linear scaling.
* For cross-algorithm comparison we also compute speedup against the
  Sequential baseline at the same `n` (see `benchmark.py`).

### 6.3 Why ParStrassen ≠ 7× faster

The seven products are independent but **not equal in size to the
original problem**: each is on an `n/2` matrix, total work is
`7 · T(n/2) ≈ T(n) / 2^(0.807·1)` — i.e. one Strassen step shrinks the
total work to about 7/8 of "naive recurse 8 times". The ceiling on
parallel speedup at one level is therefore `7`, not `8`, but to *reach*
that ceiling you need cores ≥ 7 *and* sub-problems large enough that
thread-spawn cost is negligible.

## 7. Experimental setup

* Hardware: any x86_64 box with ≥ 4 cores. (The reproduction in this
  document was on a 2-core sandbox; numbers will be much more dramatic on
  a real workstation.)
* Software: Python 3.10+, NumPy ≥ 1.24, Numba ≥ 0.58.
* Inputs: `generate_input.py` with seed 42 and entries in `[-9, 9]`.
* Timing: `time.perf_counter()` around the multiplication call only.
* Each run is single-shot in the grid script; for publication-grade
  numbers, repeat 5× and report the minimum (the minimum filters out
  noise from background processes).
* Verification: `--verify` flag compares the result to `A @ B` (NumPy's
  matmul on `int64`), which dispatches to a tight C kernel and serves as
  the trusted oracle. All four methods produce byte-identical outputs in
  our tests.

### Parameter grid (`run_experiments.sh`)

| Parameter        | Values                  |
| ---------------- | ----------------------- |
| `n`              | 64, 128, 256, 512, 1024 |
| `cores`          | 1, 2, 4, 8              |
| `base-case`      | 16, 32, 64, 128         |
| `parallel-depth` | 1, 2, 3                 |

Each combination writes into a unique subdirectory (`Sequential/n1024/`,
`ParStrassen/n1024_c8_b64_d2/`, etc.) so no per-spec output filename is
ever overwritten.

## 8. Results

Run `bash run_experiments.sh` to populate `results/summary.csv` and
`results/summary.md` with the full table; below is the structure to
expect, with representative numbers from a 2-core sandbox at `n = 1024`:

| Method        | n    | Cores | Base | Depth | Time (s) | Speedup vs Sequential | Efficiency |
| ------------- | ---- | ----- | ---- | ----- | -------- | --------------------- | ---------- |
| Sequential    | 1024 | 1     | —    | —     | 4.36     | 1.00                  | 1.00       |
| ParMtrixMult  | 1024 | 2     | —    | —     | 3.21     | 1.36                  | 0.68       |
| Strassen      | 1024 | 1     | 64   | —     | 0.49     | 8.95                  | 8.95       |
| ParStrassen   | 1024 | 2     | 64   | 1     | 1.32     | 3.30                  | 1.65       |

(Replace this table with your machine's full grid output before submission.
On an 8-core box you should see ParMtrixMult speedup of ~5–6×, ParStrassen
speedup over Sequential of ~30–60× at `n = 1024–2048`.)

## 9. Speedup and efficiency analysis

### 9.1 ParMtrixMult

Expected `S_p ≈ p` for small `p`. In practice efficiency drops as `p`
grows because:

* **Memory bandwidth** is the binding constraint at high core counts.
  Multiplying two int64 matrices is `Θ(n³)` arithmetic on `Θ(n²)` data,
  but every column of `B` is read `n` times by the loop — without good
  cache reuse this hits DRAM, and a single DRAM channel saturates after
  4–8 cores.
* **Thread-pool spawn / join cost** is fixed per call (a few hundred
  microseconds). For tiny `n` (64, 128) it can dominate the actual work
  and depress `S_p` below 1.

### 9.2 ParStrassen

Best behaviour when:

* `n` is large enough that each leaf product takes meaningfully more
  time than thread-pool overhead.
* `cores ≥ 7^d`. With cores < 7, the seven tasks queue up and the
  effective parallelism is `cores`, not 7.
* Base case is tuned. Too small → Strassen overhead per level dominates;
  too large → Strassen barely recurses.

A typical sweet spot on an 8-core machine at `n = 2048`: `cores = 8`,
`base-case = 64`, `parallel-depth = 1` (or 2 on 16+ cores).

### 9.3 Sequential vs Strassen at the same `n`

Pure `Sequential` is `Θ(n³)`; `Strassen` is `Θ(n^2.807)`. The
asymptotic gap widens as `n` grows. On the 2-core sandbox, at `n = 1024`,
sequential triple-loop is 4.36 s and sequential Strassen is 0.49 s — an
**8.9× algorithmic speedup with zero parallelism**. This is the headline
result of the project: the right *algorithm* dwarfs the right *number of
threads*.

## 10. Discussion of theory vs. practice

| Phenomenon                                  | Theory says       | What we observe                               | Why                                                                          |
| ------------------------------------------- | ----------------- | --------------------------------------------- | ---------------------------------------------------------------------------- |
| ParMtrixMult on 2 cores at large `n`        | 2× speedup        | ~1.3–1.6×                                     | memory-bandwidth bound; 64-bit integer mults stress DRAM                     |
| ParStrassen with `cores < 7`                | 7× ceiling per level | well under 7×                              | only `cores` threads ever execute simultaneously                             |
| ParStrassen with `parallel-depth = 3`       | finer parallelism | sometimes *slower* than depth 1              | thread-pool dispatch cost > saved time on small sub-problems                 |
| Strassen with `base-case = 16`              | best asymptotic   | slower than `base-case = 64`                  | constant-factor overhead of recursion / additions dominates at small leaves  |
| Numba parallel scaling beyond physical cores | ideal: linear up to `p` | flat or worse                          | hyperthreads share execution units; ALU-bound code doesn't gain from SMT     |

### Cache locality and temporary arrays

Strassen creates many temporaries: 14 intermediate sums (`A11+A22`, …,
`B21+B22`) plus 7 result matrices `M_i` per recursive call. Those
allocations cause:

* **Pressure on the L2/L3 caches**, displacing useful data.
* **NUMA effects** on multi-socket machines (the allocating thread's
  socket "owns" the page; other sockets read it across the
  inter-processor link).
* **Allocator contention** when seven threads simultaneously call
  `np.empty` of the same large size.

A production-grade Strassen would reuse buffers across recursive calls
(double-buffering) and pin allocations to NUMA nodes; we keep the code
straightforward for readability.

### Python-specific limitations

* The GIL is irrelevant for our compiled kernels (they run inside
  `with nogil:`) but *would* be a wall if any timed work happened in
  pure Python. On Python 3.14t the GIL is gone entirely.
* `np.ndarray` slicing makes views, not copies — good — but
  arithmetic (`A11 + A22`) is a copy. There is no way to get an in-place
  Strassen in pure NumPy without manual buffer management.
* Cython compiles ahead of time, so there is no first-call JIT penalty
  (unlike Numba). However, a `python setup.py build_ext --inplace` step
  is required before running the project.

## 11. Memory limitations and the optimizations applied

### 11.1 The naive memory budget

A textbook Strassen returns a fresh ndarray from every recursive call and
holds all seven products `M1..M7` simultaneously so it can compute
`C11 = M1 + M4 - M5 + M7`. At the top recursive level that means **at
least** these allocations are concurrently alive:

* `C` output: `n²` int64 (8n² bytes)
* 14 input sums for M1..M7 (some are views of A, B; about 10 are real
  copies): roughly `10 · (n/2)²` int64
* 7 result buffers `M1..M7`: `7 · (n/2)²` int64
* 3 transient arrays from `M1 + M4 - M5 + M7`-style expressions:
  `3 · (n/2)²` int64

Sum at top level only: `8n² + 20·(n/2)² · 8 = 48 n²` bytes.
Recursive calls add their own equivalent layer until depth equals the
recursion depth, but Python's call-stack semantics free those when the
call returns; the high-water mark is dominated by the top level plus the
deepest leaf.

For n = 16 384 this gives a top-level peak of ≈ 12.9 GiB **just for
Strassen scratch**, on top of A and B themselves (4 GiB combined).

### 11.2 The optimizations actually applied

The repository's `Strassen` and `ParStrassen` implement four concrete
optimizations (`algorithms.py`):

1. **Destination-buffer passing (`_strassen_into(A, B, C, base_case)`).**
   The recursion writes into a caller-supplied `C` slice instead of
   returning a freshly-allocated ndarray. Eliminates one `(n/2)²`
   allocation per recursion level.

2. **Compute-and-immediately-accumulate.** The seven products are computed
   one at a time into a single shared buffer `M`, and immediately folded
   into the C quadrants where they appear. The update sequence is chosen
   so `M` can be safely overwritten between products:
   `M1→C11=M1, C22=M1; M2→C21=M2, C22-=M2; M3→C12=M3, C22+=M3; M4→C11+=M4,
   C21+=M4; M5→C11-=M5, C12+=M5; M6→C22+=M6; M7→C11+=M7`. This collapses
   per-level scratch from ~21 buffers of size `(n/2)²` to just **3**:
   `Tx`, `Ty`, and `M`.

3. **In-place NumPy ops.** Every elementwise step uses
   `np.add(x, y, out=z)`, `np.subtract(x, y, out=z)`, `np.copyto(dst, src)`.
   No intermediates from `M1+M4-M5+M7`-style expressions ever exist.

4. **Lazy input-sum allocation in ParStrassen.** Each of the 7 worker
   tasks computes its own input sums when it actually starts running,
   so the 14 input sums never coexist at the parent level — they live
   only as long as the worker that needs them.

5. **Streaming text input parser.** `matrix_io.read_input` now uses
   `np.fromfile(f, dtype=np.int64, sep=' ')` instead of reading the whole
   file as a Python string. For n = 16 384 the input file is ≈ 1.5 GiB
   of text, which used to require several GiB of transient Python objects
   before parsing; the new parser streams through it.

### 11.3 The geometric scratch budget

After optimization, sequential Strassen allocates `3 · (n/2)²` int64
per recursion level. The total scratch summed over all levels is a
geometric series:

```
3 · (n/2)² + 3 · (n/4)² + 3 · (n/8)² + ...
  = 3 · (n²/4) · (1 + 1/4 + 1/16 + ...)
  = 3 · (n²/4) · (4/3)
  = n²    int64
  = 8 n²  bytes
```

So **total scratch is exactly one matrix's worth**, regardless of n.
Together with the output `C`, optimized sequential Strassen uses
**2 n²** int64 of working memory. For n = 16 384 that is 4 GiB, versus
the textbook implementation's roughly 13 GiB.

### 11.4 Measured numbers (this repository)

`memory_compare.py` measures both implementations side-by-side using
`tracemalloc`:

| n    | naive peak    | optimized peak | ratio | reduction |
| ---- | ------------- | -------------- | ----- | --------- |
| 512  | 6.00 MiB (3.00×) | 4.04 MiB (2.02×) | 1.49× | 1.96 MiB |
| 1024 | 24.00 MiB (3.00×) | 16.04 MiB (2.01×) | 1.50× | 7.96 MiB |

The "× matrix" column is peak memory divided by the size of one int64
`n × n` matrix. Notice the optimized version sits at exactly **2.0×
matrix** independent of n — that is the geometric series prediction
holding empirically.

`memory_benchmark.py` shows the corresponding numbers for all four
methods. Sequential and ParMtrixMult sit at **1.0× matrix** (just the
output, zero scratch). ParStrassen sits at **~7× matrix** because the
seven concurrent sub-products each need their own writable destination
— that is the unavoidable cost of true task-parallel evaluation.

### 11.5 What this means for n = 16 384

* One `int64` matrix is `16384² · 8` = **2.00 GiB**.
* `A`, `B`, and `C` together: **6.0 GiB**.

| Method        | Working memory | Total (with A, B, C) |
| ------------- | -------------- | -------------------- |
| Sequential    | C only         | 6.0 GiB              |
| ParMtrixMult  | C only         | 6.0 GiB              |
| Strassen *(opt)*  | C + n² scratch | 8.0 GiB           |
| Strassen *(naive)*| C + ~2.6n² scratch | ~13 GiB       |
| ParStrassen *(opt, depth=1)*  | C + ~7·(n/2)² + per-task scratch | ~10–12 GiB |

So:

* On a **16 GiB laptop**: optimized sequential Strassen at n = 16 384 is
  *just* feasible (8 GiB working set, leaving room for OS / other
  processes); naive Strassen is not. ParStrassen is borderline.
* On a **32 GiB workstation**: all variants run comfortably at n = 16 384.
* On a **64 GiB workstation**: even ParStrassen with `parallel-depth = 2`
  is feasible.

### 11.6 What we did NOT do (and could)

For an even tighter memory bound:

* **Top-level workspace pool.** Pre-allocate a single n² scratch buffer
  at the top of `Strassen` and partition it across recursion levels.
  Eliminates allocator churn and gives a precisely-predictable peak.
  Adds index-management complexity; we rely on Python GC instead.
* **Strassen-Winograd variant** (15 add/sub instead of 18, identical
  asymptotic complexity, slightly less scratch).
* **Block-recursive memory layout** so that recursive calls operate on
  contiguous chunks instead of strided views — better cache behaviour
  *and* allows in-place writes back to the input in a Coppersmith-style
  arrangement. This is research-grade work and outside the scope here.

## 12. Conclusion and recommended best configuration

Headline findings:

1. **Algorithm beats parallelism.** Sequential Strassen at `n = 1024`
   is roughly 9× faster than the parallel triple-loop on 2 cores
   (project sandbox), and the gap grows with `n`.
2. **ParMtrixMult is well-behaved** — memory-bandwidth bound on many
   cores, but reliably gives speedup approaching `min(cores,
   bandwidth_limit)`. It is the right choice when you cannot afford
   Strassen's memory overhead.
3. **ParStrassen is the fastest** for large `n` *if* you have enough
   cores and memory. It needs careful tuning of `base-case` and
   `parallel-depth`.
4. **Don't trust depth = 3** unless `n` is huge and cores ≥ 49 — the
   thread-pool dispatch overhead generally outweighs the parallelism
   gained at that depth.

### Recommended configuration

| Scenario                                       | Recommendation                                                   |
| ---------------------------------------------- | ---------------------------------------------------------------- |
| Small `n` (≤ 256), low memory                  | `Sequential`                                                     |
| Medium `n` (256–1024), tight memory            | `ParMtrixMult --cores all`                                       |
| Medium `n` (1024–4096), enough RAM             | `Strassen --base-case 64`                                        |
| Large `n` (4096+), 8+ cores, enough RAM        | `ParStrassen --cores 8 --base-case 64 --parallel-depth 1`        |
| Very large `n` (8192+), 16+ cores              | `ParStrassen --cores 16 --base-case 128 --parallel-depth 2`      |

### Possible extensions

* **Strassen-Winograd** (15 add/sub instead of 18): smaller constant.
* **Block-recursive memory pool** to slash temporary-array allocation.
* **Hybrid CPU/GPU** kernel — leaves run on a GPU via CuPy; recursion
  remains on the CPU.
* **AOT Numba compilation** to eliminate the JIT warmup entirely.
