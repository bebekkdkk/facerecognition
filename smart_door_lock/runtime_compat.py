"""
Runtime compatibility guardrails.

This module should be imported before numpy/cv2/tflite imports in entrypoints.
It only applies conservative thread limits on Raspberry Pi/ARM devices so
desktop environments are not unnecessarily constrained.
"""

import os
import platform


def _is_low_power_arm_target():
    """Return True for Raspberry Pi-like ARM environments."""
    machine = platform.machine().lower()
    is_arm = machine.startswith("arm") or machine.startswith("aarch")
    # /proc/device-tree/model only exists on Linux SBCs like Raspberry Pi.
    return is_arm and os.path.exists("/proc/device-tree/model")


def apply_runtime_guards():
    """Apply conservative runtime settings only where they are needed."""
    if not _is_low_power_arm_target():
        return

    # Keep BLAS/OpenMP single-threaded to reduce CPU pressure on Pi 3.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    # Prefer portable OpenBLAS core profile for 32-bit ARM builds.
    if platform.machine().lower().startswith("armv7"):
        os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV7")


apply_runtime_guards()
