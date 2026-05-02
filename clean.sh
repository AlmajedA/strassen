#!/usr/bin/env bash
# =============================================================================
# clean.sh
# =============================================================================
# Remove generated artifacts from the project directory.
#
# By default (no flags) deletes:
#   - results/                  the experiment output directory
#   - build/                    Cython build cache
#   - algorithms_omp.c          Cython-generated C source
#   - algorithms_omp.*.so       compiled extension (Linux/macOS)
#   - algorithms_omp.*.pyd      compiled extension (Windows)
#   - __pycache__/              Python bytecode caches (recursive)
#   - *.pyc                     stray bytecode files (recursive)
#
# Optional flags:
#   --inputs       also delete the generated input files in inputs/
#   --purge        also delete results/dashboard.html, summary.csv, summary.md
#                  (default: these are kept so you don't lose final reports)
#   --all          everything: --inputs and --purge combined
#   -y, --yes      skip the confirmation prompt
#   -h, --help     show this help
#
# Usage examples:
#   bash clean.sh                  # interactive, keeps inputs and dashboard
#   bash clean.sh -y               # no prompt, keeps inputs and dashboard
#   bash clean.sh --purge          # also delete dashboard / summaries
#   bash clean.sh --all -y         # nuke everything, no prompt
# =============================================================================

set -e

# ---------- parse args ----------
DELETE_INPUTS=0
PURGE=0
ASSUME_YES=0

while [ $# -gt 0 ]; do
    case "$1" in
        --inputs)    DELETE_INPUTS=1 ;;
        --purge)     PURGE=1 ;;
        --all)       DELETE_INPUTS=1; PURGE=1 ;;
        -y|--yes)    ASSUME_YES=1 ;;
        -h|--help)
            sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            echo "use -h for help" >&2
            exit 2
            ;;
    esac
    shift
done

# ---------- gather targets ----------
TARGETS=()
[ -d results ]                 && TARGETS+=("results/")
[ -d build ]                   && TARGETS+=("build/")
[ -f algorithms_omp.c ]        && TARGETS+=("algorithms_omp.c")

# Compiled extension (glob may not match — only add if it does).
for f in algorithms_omp*.so algorithms_omp*.pyd; do
    [ -e "$f" ] && TARGETS+=("$f")
done

# Python caches found anywhere in the tree.
PYCACHE_DIRS=$(find . -type d -name "__pycache__" 2>/dev/null || true)
PYC_FILES=$(find . -type f -name "*.pyc" 2>/dev/null || true)
[ -n "$PYCACHE_DIRS" ] && TARGETS+=("__pycache__ dirs ($(echo "$PYCACHE_DIRS" | wc -l))")
[ -n "$PYC_FILES" ]    && TARGETS+=("*.pyc files ($(echo "$PYC_FILES" | wc -l))")

if [ "$DELETE_INPUTS" -eq 1 ] && [ -d inputs ]; then
    INPUT_COUNT=$(find inputs -maxdepth 1 -name "*.txt" -type f 2>/dev/null | wc -l)
    [ "$INPUT_COUNT" -gt 0 ] && TARGETS+=("inputs/*.txt ($INPUT_COUNT files)")
fi

# ---------- nothing to do? ----------
if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "Nothing to clean — workspace is already tidy."
    exit 0
fi

# ---------- show plan ----------
echo "About to delete:"
for t in "${TARGETS[@]}"; do
    echo "  - $t"
done
echo ""

# ---------- confirm ----------
if [ "$ASSUME_YES" -ne 1 ]; then
    read -r -p "Proceed? [y/N] " reply
    case "$reply" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "Aborted."; exit 1 ;;
    esac
fi

# ---------- delete ----------

# Preserve final deliverables that live inside results/ (dashboard, summaries),
# UNLESS --purge was passed.
PRESERVED=()
if [ -d results ] && [ "$PURGE" -ne 1 ]; then
    TMP_PRESERVE=$(mktemp -d)
    for keep in dashboard.html summary.csv summary.md; do
        if [ -f "results/$keep" ]; then
            mv "results/$keep" "$TMP_PRESERVE/$keep"
            PRESERVED+=("$keep")
        fi
    done
fi

rm -rf results build algorithms_omp.c
rm -f  algorithms_omp*.so algorithms_omp*.pyd

# Restore preserved files.
if [ ${#PRESERVED[@]} -gt 0 ]; then
    mkdir -p results
    for keep in "${PRESERVED[@]}"; do
        mv "$TMP_PRESERVE/$keep" "results/$keep"
    done
    rmdir "$TMP_PRESERVE"
    echo "Kept: ${PRESERVED[*]} (in results/)"
elif [ -n "${TMP_PRESERVE:-}" ]; then
    rmdir "$TMP_PRESERVE" 2>/dev/null || true
fi

# Python caches.
[ -n "$PYCACHE_DIRS" ] && find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
[ -n "$PYC_FILES" ]    && find . -type f -name "*.pyc"       -delete       2>/dev/null || true

# Generated inputs (only if requested).
if [ "$DELETE_INPUTS" -eq 1 ] && [ -d inputs ]; then
    find inputs -maxdepth 1 -name "*.txt" -type f -delete 2>/dev/null || true
fi

echo "Done."