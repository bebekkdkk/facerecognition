"""Utility helpers for loading TensorFlow Lite interpreter safely."""


def get_tflite_interpreter_class():
    """
    Return TensorFlow Lite Interpreter class with lightweight-first import order.

    Priority:
    1) tflite_runtime.interpreter.Interpreter
    2) tensorflow.lite.Interpreter
    """
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        return Interpreter
    except Exception:
        pass

    try:
        from tensorflow.lite import Interpreter  # type: ignore
        return Interpreter
    except Exception:
        pass

    raise ImportError(
        "No TensorFlow Lite interpreter available. Install `tflite-runtime` "
        "for Raspberry Pi 3 (recommended)."
    )
