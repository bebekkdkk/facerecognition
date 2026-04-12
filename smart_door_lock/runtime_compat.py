
"""
Runtime compatibility guardrails.

This module should be imported before numpy/cv2/tflite imports in entrypoints.
It applies conservative thread/BLAS limits only for older, low-power ARM
targets (e.g. Raspberry Pi 2/3). Raspberry Pi 4 (aarch64) is not constrained
here to allow higher throughput on the more capable hardware.
"""

import os
import platform


def _is_low_power_arm_target():
    """Return True for older, low-power ARM targets (Raspberry Pi 2/3).

    We intentionally avoid treating Raspberry Pi 4 (aarch64) as a low-power
    ARM target because its performance characteristics differ from Pi3.
    """
    machine = platform.machine().lower()
    # Only consider 32-bit ARM variants as low-power (armv6/armv7).
    is_lowpower_arm = machine.startswith("armv7") or machine.startswith("armv6")
    return is_lowpower_arm and os.path.exists("/proc/device-tree/model")


def apply_runtime_guards():
    """Apply conservative runtime settings only where they are needed."""
    if not _is_low_power_arm_target():
        return

    # Keep BLAS/OpenMP single-threaded to reduce CPU pressure on older Pi.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    # Prefer portable OpenBLAS core profile for 32-bit ARM builds.
    if platform.machine().lower().startswith("armv7"):
        os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV7")


apply_runtime_guards()
