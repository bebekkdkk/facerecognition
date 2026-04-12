"""
Face Anti-Spoofing Module
- Detect live vs spoof using a small TFLite model and Laplacian blur check.

Behavior follows the IMPLEMENTATION_SUMMARY requirements:
- Resize to 256x256, normalize, Laplacian-based blur rejection,
- Run TFLite model and compute score = sum(abs(pred) * mask),
- Threshold 0.2: score < 0.2 => REAL.
"""

import threading
import cv2
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.tflite_utils import get_tflite_interpreter_class
from config import ANTI_SPOOF_INPUT_SIZE, ANTI_SPOOF_THRESHOLD, ANTI_SPOOF_LAPLACE_THRESHOLD

# Module-level interpreter lock for thread-safety
_tflite_interpreter = None
_tflite_lock = threading.Lock()


class FaceAntiSpoofing:
    """Anti-spoofing detector using a TFLite model with Laplacian pre-check."""

    INPUT_IMAGE_SIZE = ANTI_SPOOF_INPUT_SIZE
    THRESHOLD = ANTI_SPOOF_THRESHOLD
    LAPLACE_THRESHOLD = ANTI_SPOOF_LAPLACE_THRESHOLD

    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._init_interpreter()

    def _init_interpreter(self):
        Interpreter = get_tflite_interpreter_class()
        with _tflite_lock:
            try:
                self.interpreter = Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except Exception as e:
                raise RuntimeError(f"Failed to load anti-spoofing model: {e}")

    @staticmethod
    def _calculate_laplacian(image_gray):
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        lap = cv2.filter2D(image_gray, cv2.CV_64F, kernel)
        return float(np.var(lap))

    def _check_blur(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        lap_var = self._calculate_laplacian(gray)
        is_sharp = lap_var >= self.LAPLACE_THRESHOLD
        return is_sharp, lap_var

    def detect(self, face_image):
        if self.interpreter is None:
            raise RuntimeError("Interpreter not initialized")

        # 1) Blur check
        is_sharp, lap_var = self._check_blur(face_image)
        if not is_sharp:
            return {
                'is_real': False,
                'score': 1.0,
                'laplacian_variance': lap_var,
                'label': 'FAKE',
                'confidence': 1.0,
                'reason': 'Blur detected - image too blurry'
            }

        # 2) Preprocess for model
        img = cv2.resize(face_image, (self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img, axis=0)

        input_detail = self.input_details[0]
        if input_detail['dtype'] == np.uint8:
            scale, zero_point = input_detail.get('quantization', (0.0, 0))
            if scale and scale > 0:
                input_data = (input_data / scale + zero_point).astype(np.uint8)
            else:
                input_data = (input_data * 255.0).astype(np.uint8)
        else:
            input_data = input_data.astype(np.float32)

        # 3) Inference
        with _tflite_lock:
            try:
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output = self.interpreter.get_tensor(self.output_details[0]['index'])
            except Exception as e:
                raise RuntimeError(f"Inference failed: {e}")

        # 4) Dequantize if needed
        out_detail = self.output_details[0]
        if out_detail['dtype'] == np.uint8:
            out_scale, out_zero = out_detail.get('quantization', (0.0, 0))
            if out_scale and out_scale > 0:
                output = (output.astype(np.float32) - out_zero) * out_scale
            else:
                output = output.astype(np.float32)

        clss_pred = output.flatten()

        if clss_pred.size > 0:
            leaf_node_mask = np.ones_like(clss_pred)
            score = float(np.sum(np.abs(clss_pred) * leaf_node_mask))
        else:
            score = 0.0

        # Normalize and threshold
        if score > 1.0 and np.sum(np.abs(clss_pred)) > 0:
            score = score / float(np.sum(np.abs(clss_pred)))

        is_real = score < self.THRESHOLD

        return {
            'is_real': bool(is_real),
            'score': float(score),
            'laplacian_variance': float(lap_var),
            'label': 'REAL' if is_real else 'FAKE',
            'confidence': float(min(1.0 - score if is_real else score, 1.0)),
            'raw_output': clss_pred.tolist()
        }


class AntiSpoofingPipeline:
    def __init__(self, model_path):
        self.detector = FaceAntiSpoofing(model_path)

    def process(self, face_image):
        return self.detector.detect(face_image)
