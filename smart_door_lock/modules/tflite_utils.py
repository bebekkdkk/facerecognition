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
        import tensorflow as tf  # type: ignore
        interpreter_cls = getattr(tf.lite, "Interpreter", None)
        if interpreter_cls is not None:
            return interpreter_cls
    except Exception:
        pass

    raise ImportError(
        "No TensorFlow Lite interpreter available. Install `tensorflow-cpu` "
        "for desktop (Windows/Linux/macOS) or `tflite-runtime` for Raspberry Pi."
    )
