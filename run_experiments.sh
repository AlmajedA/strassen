#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh
# =============================================================================
# Drive the experiment grid for Strassen and ParStrassen:
#   - generate input matrices for several n
#   - sweep base-case / cores / parallel-depth
#   - verify correctness
#   - aggregate everything into a summary CSV + Markdown table
# =============================================================================

set -e

# ---------- configuration ----------
INPUT_DIR="${INPUT_DIR:-inputs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PYTHON="${PYTHON:-python}"

# Activate a virtual environment if one exists (optional).
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Test sizes (powers of 2). Add 2048, 4096 if your machine can take it.
SIZES=(64 128 256 512 1024)

# Which methods to run. Comment out any you want to skip.
METHODS=(Strassen ParStrassen)

# Thread counts to sweep for ParStrassen.
CORES=(1 2 4 8)

# Strassen base-case thresholds to sweep.
BASES=(16 32 64 128)

# Parallel depths to sweep for ParStrassen.
DEPTHS=(1 2 3)

# Whether to verify against NumPy's matmul (slow for large n; toggle as needed).
VERIFY="${VERIFY:---verify}"

# Cap large-n verification (numpy oracle gets slow on huge int64 mats too).
VERIFY_MAX_N="${VERIFY_MAX_N:-1024}"

# ---------- build Cython+OpenMP extension ----------
echo "Building Cython+OpenMP kernels..."
"$PYTHON" setup.py build_ext --inplace || { echo "FATAL: Cython build failed"; exit 1; }
echo ""

# ---------- prep ----------
mkdir -p "$INPUT_DIR" "$RESULTS_DIR"

echo "==========================================================="
echo "Strassen experiment runner"
echo "  methods: ${METHODS[*]}"
echo "  sizes:   ${SIZES[*]}"
echo "  cores:   ${CORES[*]}"
echo "  bases:   ${BASES[*]}"
echo "  depths:  ${DEPTHS[*]}"
echo "  inputs:  $INPUT_DIR"
echo "  outputs: $RESULTS_DIR"
echo "==========================================================="

# ---------- generate inputs ----------
for n in "${SIZES[@]}"; do
    f="$INPUT_DIR/input_${n}.txt"
    if [ ! -f "$f" ]; then
        echo "[gen] $f"
        "$PYTHON" generate_input.py -n "$n" -o "$f"
    else
        echo "[gen] $f already exists, skipping"
    fi
done

# Helper: decide whether to pass --verify for a given n.
verify_flag_for() {
    local n="$1"
    if [ -n "$VERIFY" ] && [ "$n" -le "$VERIFY_MAX_N" ]; then
        echo "$VERIFY"
    else
        echo ""
    fi
}

# Helper: check if a method is in the METHODS array.
method_enabled() {
    local target="$1"
    for m in "${METHODS[@]}"; do
        if [ "$m" = "$target" ]; then return 0; fi
    done
    return 1
}

run_one() {
    local subdir="$1"; shift
    mkdir -p "$RESULTS_DIR/$subdir"
    echo "[run] $subdir :: $*"
    "$PYTHON" main.py "$@" --output-dir "$RESULTS_DIR/$subdir" || \
        { echo "[FAIL] $subdir"; return 1; }
}

# ---------- 1. Sequential Strassen, sweep base-case ----------
if method_enabled Strassen; then
for n in "${SIZES[@]}"; do
    inp="$INPUT_DIR/input_${n}.txt"
    vf=$(verify_flag_for "$n")
    for b in "${BASES[@]}"; do
        if [ "$b" -ge "$n" ]; then continue; fi
        run_one "Strassen/n${n}_b${b}" --input "$inp" --method Strassen \
                --base-case "$b" $vf
    done
done
else echo "[skip] Strassen"; fi

# ---------- 2. Parallel Strassen, sweep cores x base-case x depth ----------
if method_enabled ParStrassen; then
for n in "${SIZES[@]}"; do
    inp="$INPUT_DIR/input_${n}.txt"
    vf=$(verify_flag_for "$n")
    for c in "${CORES[@]}"; do
        for b in "${BASES[@]}"; do
            if [ "$b" -ge "$n" ]; then continue; fi
            for d in "${DEPTHS[@]}"; do
                run_one "ParStrassen/n${n}_c${c}_b${b}_d${d}" \
                        --input "$inp" --method ParStrassen \
                        --cores "$c" --base-case "$b" --parallel-depth "$d" $vf
            done
        done
    done
done
else echo "[skip] ParStrassen"; fi

# ---------- 3. Aggregate ----------
echo "==========================================================="
echo "Aggregating results..."
"$PYTHON" benchmark.py "$RESULTS_DIR"
echo "Done. See $RESULTS_DIR/summary.csv and $RESULTS_DIR/summary.md"
echo "==========================================================="