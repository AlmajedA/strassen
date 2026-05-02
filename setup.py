"""
setup.py
========

Build the Cython + OpenMP extension module ``algorithms_omp``.

    python setup.py build_ext --inplace

This produces ``algorithms_omp.*.so`` (Linux) or ``algorithms_omp.*.pyd``
(Windows) in the current directory, which ``algorithms.py`` imports.

Compiler flags
--------------
GCC / Clang:  -fopenmp  (compile AND link)
MSVC:         /openmp    (compile only; link flag not needed)

We default to the GCC/Clang flags.  If you're on MSVC, replace
``-fopenmp`` with ``/openmp`` in extra_compile_args and remove
extra_link_args.
"""

import os
import sys
from setuptools import Extension, setup
from Cython.Build import cythonize

# Platform-aware OpenMP and optimization flags.
if sys.platform == "win32":
    omp_compile = ["/openmp"]
    omp_link = []
    opt_flags = ["/O2"]
else:
    omp_compile = ["-fopenmp"]
    omp_link = ["-fopenmp"]
    opt_flags = ["-O3"]

ext_modules = [
    Extension(
        "algorithms_omp",
        sources=["algorithms_omp.pyx"],
        extra_compile_args=opt_flags + omp_compile,
        extra_link_args=omp_link,
    )
]

setup(
    name="strassen-omp",
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
        },
    ),
)