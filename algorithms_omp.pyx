# cython: boundscheck=False, wraparound=False, cdivision=True
"""
algorithms_omp.pyx
==================

Low-level matrix-multiplication kernel compiled to C with OpenMP support.

Exported functions
------------------
seq_kernel(A, B, C)
    Single-threaded O(n^3) triple loop.  Releases the GIL while running so
    that other Python threads (e.g. the 7 Strassen tasks dispatched by
    ThreadPoolExecutor) can execute concurrently on different cores.

    Used as the base-case multiplier by both Strassen and ParStrassen.

get_max_threads()
    Thin wrapper around ``omp_get_max_threads()`` so Python code can
    query the OpenMP default thread count for info-file reporting.

Build
-----
    python setup.py build_ext --inplace

This compiles with ``-fopenmp`` (GCC/Clang) and links against libgomp.
"""

cimport openmp


# --------------------------------------------------------------------------- #
# Sequential kernel (base case for Strassen recursion)                        #
# --------------------------------------------------------------------------- #

def seq_kernel(long long[:, :] A, long long[:, :] B, long long[:, :] C):
    """Write A @ B into C.  Single-threaded; releases the GIL."""
    cdef Py_ssize_t n = A.shape[0]
    cdef Py_ssize_t p = A.shape[1]
    cdef Py_ssize_t m = B.shape[1]
    cdef Py_ssize_t i, j, k
    cdef long long acc

    with nogil:
        for i in range(n):
            for j in range(m):
                acc = 0
                for k in range(p):
                    acc = acc + A[i, k] * B[k, j]
                C[i, j] = acc


# --------------------------------------------------------------------------- #
# Helper                                                                      #
# --------------------------------------------------------------------------- #

def get_max_threads():
    """Return the current OpenMP max-thread setting (omp_get_max_threads)."""
    return openmp.omp_get_max_threads()
