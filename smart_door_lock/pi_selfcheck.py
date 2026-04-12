"""
Raspberry Pi 3 compatibility self-check.

Run on Raspberry Pi before starting app:
    python3.9 pi_selfcheck.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_import_check(module_name):
    cmd = [sys.executable, "-c", f"import {module_name}; print('OK')"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def print_result(title, ok, detail=""):
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {title}")
    if detail:
        print(f"      {detail}")


def main():
    if not os.path.exists('/proc/device-tree/model'):
        print("This self-check is intended for Raspberry Pi.")
        print(f"Detected platform: {platform.system()} {platform.machine()}")
        print("Skipping Raspberry Pi compatibility checks.")
        return 0

    print("=" * 60)
    print("Raspberry Pi 3 Runtime Self-Check")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {sys.platform}")

    checks = [
        ("numpy", "NumPy import"),
        ("cv2", "OpenCV import"),
        ("tflite_runtime.interpreter", "TFLite runtime import"),
    ]

    all_ok = True
    for module_name, label in checks:
        code, out, err = run_import_check(module_name)
        if code == 0:
            print_result(label, True)
        else:
            all_ok = False
            detail = err or out or f"return code={code}"
            if code in (132, -4):
                detail = (
                    "Detected SIGILL (illegal instruction). Reinstall this package "
                    "with Raspberry Pi compatible wheel/apt package."
                )
            print_result(label, False, detail)

    base = Path(__file__).resolve().parent
    models = [
        base / "models" / "MobileFaceNet.tflite",
        base / "models" / "FaceAntiSpoofing.tflite",
        base / "models" / "haarcascade_frontalface_default.xml",
    ]

    for model_path in models:
        print_result(f"Model exists: {model_path.name}", model_path.exists())
        all_ok = all_ok and model_path.exists()

    print("=" * 60)
    if all_ok:
        print("Self-check PASSED. You can run enrollment.py/main.py")
        return 0

    print("Self-check FAILED. Fix failed items first to avoid runtime crash.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
