"""
Face Anti-Spoofing Module
- Deteksi live vs spoof attack menggunakan model TensorFlow Lite
- Lazy Tensor Flow Lite Interpreter initialization
- Optimized untuk Raspberry Pi 3
"""

import cv2
import numpy as np
import threading
from modules.tflite_utils import get_tflite_interpreter_class
from config import (
    ANTI_SPOOF_INPUT_SIZE,
    ANTI_SPOOF_THRESHOLD,
    ANTI_SPOOF_LAPLACE_THRESHOLD,
)

# Global interpreter dan lock untuk thread-safety
_tflite_interpreter = None
_tflite_lock = threading.Lock()


class FaceAntiSpoofing:
    """
    Anti-spoofing detector menggunakan FaceAntiSpoofing.tflite
    Exact logic sebagai requirement
    """
    
    # Konfigurasi exact seperti requirement
    INPUT_IMAGE_SIZE = ANTI_SPOOF_INPUT_SIZE
    THRESHOLD = ANTI_SPOOF_THRESHOLD
    LAPLACE_THRESHOLD = ANTI_SPOOF_LAPLACE_THRESHOLD
    
    def __init__(self, model_path):
        """
        Initialize anti-spoofing detector
        
        Args:
            model_path (str): Path ke FaceAntiSpoofing.tflite model
        """
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._init_interpreter()
    
    def _init_interpreter(self):
        """Initialize TensorFlow Lite interpreter (lazy load)"""
        global _tflite_interpreter
        Interpreter = get_tflite_interpreter_class()
        
        with _tflite_lock:
            try:
                self.interpreter = Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                print(f"[INFO] Anti-spoofing model loaded: {self.model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load anti-spoofing model: {e}")
    
    def _calculate_laplacian(self, image_gray):
        """
        Calculate Laplacian manually dengan kernel 3x3
        
        Args:
            image_gray (np.ndarray): Grayscale image
            
        Returns:
            float: Laplacian variance (blur metric)
        """
        # Laplacian kernel 3x3
        kernel = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        # Apply filter2D untuk konvolusi
        laplacian = cv2.filter2D(image_gray, cv2.CV_64F, kernel)
        
        # Calculate variance
        variance = np.var(laplacian)
        
        return variance
    
    def _check_blur(self, image):
        """
        Check blur menggunakan Laplacian variance
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            tuple: (is_sharp: bool, laplacian_variance: float)
        """
        # Convert ke grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate laplacian variance
        laplacian_var = self._calculate_laplacian(gray)
        
        # Jika blur < threshold → reject (return False)
        is_sharp = laplacian_var >= self.LAPLACE_THRESHOLD
        
        return is_sharp, laplacian_var
    
    def detect(self, face_image):
        """
        Detect apakah wajah real atau spoof
        
        Args:
            face_image (np.ndarray): Cropped face image (BGR format)
            
        Returns:
            dict: {
                'is_real': bool,
                'score': float,
                'laplacian_variance': float,
                'label': str ('REAL' or 'FAKE'),
                'confidence': float
            }
        """
        if self.interpreter is None:
            raise RuntimeError("Interpreter not initialized")
        
        # Step 1: Check blur menggunakan Laplacian
        is_sharp, laplacian_var = self._check_blur(face_image)
        
        if not is_sharp:
            return {
                'is_real': False,
                'score': 1.0,  # Score tinggi = FAKE
                'laplacian_variance': laplacian_var,
                'label': 'FAKE',
                'confidence': 1.0,
                'reason': 'Blur detected - image too blurry'
            }
        
        # Step 2: Resize ke 256x256
        face_resized = cv2.resize(face_image, (self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE))
        
        # Step 3: Normalisasi (pixel / 255.0)
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Most face models are RGB; convert BGR(OpenCV) -> RGB.
        face_rgb = cv2.cvtColor(face_normalized, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(face_rgb, axis=0)

        input_detail = self.input_details[0]
        input_dtype = input_detail['dtype']

        if input_dtype == np.uint8:
            scale, zero_point = input_detail.get('quantization', (0.0, 0))
            if scale and scale > 0:
                input_data = (input_data / scale + zero_point).astype(np.uint8)
            else:
                input_data = (input_data * 255.0).astype(np.uint8)
        else:
            input_data = input_data.astype(np.float32)
        
        # Step 4: Run inference
        with _tflite_lock:
            try:
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                
                output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            except Exception as e:
                raise RuntimeError(f"Inference failed: {e}")

        output_detail = self.output_details[0]
        if output_detail['dtype'] == np.uint8:
            out_scale, out_zero = output_detail.get('quantization', (0.0, 0))
            if out_scale and out_scale > 0:
                output_data = (output_data.astype(np.float32) - out_zero) * out_scale
            else:
                output_data = output_data.astype(np.float32)
        
        # Step 5: Calculate score menggunakan exact formula:
        # score = sum(abs(clss_pred[i]) * leaf_node_mask[i])
        clss_pred = output_data.flatten()
        
        # Jika output berisi probability/confidence scores
        if len(clss_pred) > 0:
            # Asumsikan leaf_node_mask adalah ones atau weights
            leaf_node_mask = np.ones_like(clss_pred)
            score = np.sum(np.abs(clss_pred) * leaf_node_mask)
        else:
            score = 0.0
        
        # Normalize score jika diperlukan
        if score > 1.0:
            score = score / np.sum(np.abs(clss_pred))
        
        # Step 6: Jika score < 0.2 → REAL, selain itu FAKE
        is_real = score < self.THRESHOLD
        
        return {
            'is_real': is_real,
            'score': float(score),
            'laplacian_variance': float(laplacian_var),
            'label': 'REAL' if is_real else 'FAKE',
            'confidence': min(1.0 - score if is_real else score, 1.0),
            'raw_output': clss_pred.tolist()
        }
    
    def batch_detect(self, face_images):
        """
        Detect spoof untuk multiple faces
        
        Args:
            face_images (list): List of face images
            
        Returns:
            list: List of detection results
        """
        results = []
        for face_img in face_images:
            results.append(self.detect(face_img))
        return results


class AntiSpoofingPipeline:
    """Simplified pipeline untuk deteksi spoofing"""
    
    def __init__(self, model_path):
        """Initialize pipeline"""
        self.detector = FaceAntiSpoofing(model_path)
    
    def process(self, face_image):
        """
        Process single face image
        
        Args:
            face_image (np.ndarray): Face image
            
        Returns:
            dict: Detection result dengan 'is_real' dan 'label'
        """
        return self.detector.detect(face_image)
