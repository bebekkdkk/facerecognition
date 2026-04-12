"""
Face Anti-Spoofing Detection menggunakan TFLite (FaceAntiSpoofing.tflite)
"""
import numpy as np
import cv2
import sys
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Warning: TFLite runtime not found. Install with: pip install tflite-runtime")
        tflite = None

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANTI_SPOOFING_MODEL, ANTI_SPOOFING


class FaceAntiSpoofing:
    """
    Deteksi spoofing attack (presentation attack) pada wajah
    Model: FaceAntiSpoofing.tflite
    """
    
    def __init__(self, model_path=ANTI_SPOOFING_MODEL):
        """
        Inisialisasi anti-spoofing model
        
        Args:
            model_path: Path ke FaceAntiSpoofing.tflite
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Anti-spoofing model not found: {model_path}")
        
        if tflite is None:
            raise RuntimeError("TFLite runtime not available. Install tflite-runtime.")
        
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_size = ANTI_SPOOFING['INPUT_IMAGE_SIZE']  # 256
        self.threshold = ANTI_SPOOFING['THRESHOLD']  # 0.2
        self.laplace_threshold = ANTI_SPOOFING['LAPLACE_THRESHOLD']  # 50
        self.laplacian_threshold = ANTI_SPOOFING['LAPLACIAN_THRESHOLD']  # 1000
    
    def calculate_laplacian(self, image):
        """
        Hitung Laplacian blur detection dengan manual 3x3 kernel
        
        Laplacian kernel:
        [  0  -1   0 ]
        [ -1   4  -1 ]
        [  0  -1   0 ]
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            Laplacian variance score
        """
        # Manual Laplacian kernel 3x3
        kernel = np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        
        # Apply kernel convolution
        laplacian = cv2.filter2D(image, cv2.CV_32F, kernel)
        variance = np.var(laplacian)
        
        return variance
    
    def preprocess_image(self, face_img):
        """
        Preprocess gambar untuk model
        
        Flow:
        1. Resize ke 256x256
        2. Convert BGR to grayscale untuk blur detection
        3. Calculate Laplacian variance
        4. Normalize (pixel / 255.0)
        
        Args:
            face_img: Input face image (BGR)
            
        Returns:
            Tuple (preprocessed_image, laplacian_score, is_blurred)
        """
        # 1. Resize ke 256x256
        resized = cv2.resize(face_img, (self.input_size, self.input_size))
        
        # 2. Konversi ke grayscale untuk blur detection
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # 3. Hitung Laplacian
        laplacian_score = self.calculate_laplacian(gray)
        is_blurred = laplacian_score < self.laplace_threshold
        
        # 4. Normalize untuk model (pixel / 255.0)
        normalized = resized.astype(np.float32) / 255.0
        
        # Expand dims untuk batch (1, 256, 256, 3)
        if len(normalized.shape) == 3:
            normalized = np.expand_dims(normalized, axis=0)
        
        return normalized, laplacian_score, is_blurred
    
    def predict(self, face_img):
        """
        Prediksi apakah wajah REAL atau FAKE
        
        Flow EXACT:
        1. Resize ke 256x256 ✓
        2. Normalize (pixel / 255.0) ✓
        3. Hitung Laplacian manual (3x3 kernel) ✓
        4. Jika blur < threshold → REJECT ✓
        5. Jika lolos → run model ✓
        6. Score = sum(abs(clss_pred[i]) * leaf_node_mask[i]) ✓
        7. Jika score < 0.2 → REAL, else FAKE ✓
        
        Args:
            face_img: Input face image
            
        Returns:
            Dict dengan keys:
                - is_real: bool (True = REAL, False = FAKE)
                - score: float (model output score)
                - laplacian_score: float (blur detection score)
                - is_blurred: bool
                - status: str ('REAL', 'FAKE', 'BLURRED', 'ERROR')
        """
        result = {
            'is_real': False,
            'score': 0.0,
            'laplacian_score': 0.0,
            'is_blurred': False,
            'status': 'UNKNOWN'
        }
        
        try:
            # Preprocess
            processed, laplacian_score, is_blurred = self.preprocess_image(face_img)
            
            result['laplacian_score'] = float(laplacian_score)
            result['is_blurred'] = bool(is_blurred)
            
            # Check blur dulu
            if is_blurred:
                result['status'] = 'BLURRED'
                return result
            
            # Run TFLite inference
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                processed.astype(self.input_details[0]['dtype'])
            )
            self.interpreter.invoke()
            
            # Get output tensor
            output_data = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            
            # Calculate score: sum(abs(pred[i]) * mask[i])
            score = np.sum(np.abs(output_data))
            
            result['score'] = float(score)
            
            # Threshold check: < 0.2 = REAL
            if score < self.threshold:
                result['is_real'] = True
                result['status'] = 'REAL'
            else:
                result['is_real'] = False
                result['status'] = 'FAKE'
            
        except Exception as e:
            print(f"Error in anti-spoofing prediction: {e}")
            result['status'] = f'ERROR: {str(e)}'
        
        return result
