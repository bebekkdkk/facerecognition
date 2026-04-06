"""
Runtime compatibility guardrails for Raspberry Pi 3.

This module should be imported before numpy/cv2/tflite imports in entrypoints
to reduce the chance of illegal instruction and oversubscription issues.
"""

import os


def apply_runtime_guards():
    """Apply conservative runtime settings for low-power ARM devices."""
    # Keep BLAS/OpenMP single-threaded to reduce CPU pressure on Pi 3.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    # Prefer portable OpenBLAS core profile if available.
    os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV7")


apply_runtime_guards()
