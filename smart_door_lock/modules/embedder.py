"""
Face Embedding Module
- Extract face embeddings menggunakan MobileFaceNet TensorFlow Lite model
- L2 normalization
- Optimized untuk Raspberry Pi 4 (Model B) dan perangkat ARM serupa
"""

import cv2
import numpy as np
import threading
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.tflite_utils import get_tflite_interpreter_class

# Global interpreter dan lock untuk thread-safety
_tflite_interpreter = None
_tflite_lock = threading.Lock()


class FaceEmbedder:
    """Face embedder menggunakan MobileFaceNet.tflite"""
    
    # Standard face size untuk MobileFaceNet
    TARGET_FACE_SIZE = (112, 112)
    EMBEDDING_DIM = 128  # MobileFaceNet typical output
    
    def __init__(self, model_path):
        """
        Initialize face embedder dengan MobileFaceNet.tflite
        
        Args:
            model_path (str): Path ke MobileFaceNet.tflite model
        """
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._init_interpreter()
    
    def _init_interpreter(self):
        """Initialize TensorFlow Lite interpreter"""
        global _tflite_interpreter
        Interpreter = get_tflite_interpreter_class()
        
        with _tflite_lock:
            try:
                self.interpreter = Interpreter(model_path=self.model_path)
                self.interpreter.allocate_tensors()
                
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                print(f"[INFO] MobileFaceNet model loaded: {self.model_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load embedder model: {e}")
    
    def preprocess_face(self, face_image):
        """
        Preprocess face image untuk MobileFaceNet
        
        Args:
            face_image (np.ndarray): Cropped face image (BGR format dari OpenCV)
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Resize ke target size
        face_resized = cv2.resize(face_image, self.TARGET_FACE_SIZE)
        
        # Convert BGR to RGB jika diperlukan
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalisasi ke [0, 1]
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Subtract mean jika diperlukan (whitening)
        # Mean values typically: [0.485, 0.456, 0.406] untuk ImageNet
        # Namun untuk face, sering menggunakan [0.5, 0.5, 0.5]
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        
        face_normalized = (face_normalized - mean) / std
        
        return face_normalized
    
    def extract_embedding(self, face_image):
        """
        Extract embedding dari face image
        
        Args:
            face_image (np.ndarray): Cropped face image
            
        Returns:
            np.ndarray: Face embedding (1D vector)
        """
        if self.interpreter is None:
            raise RuntimeError("Interpreter not initialized")
        
        # Preprocess
        face_preprocessed = self.preprocess_face(face_image)
        
        # Add batch dimension: (112, 112, 3) -> (1, 112, 112, 3)
        input_data = np.expand_dims(face_preprocessed, axis=0)
        input_detail = self.input_details[0]

        if input_detail['dtype'] == np.uint8:
            scale, zero_point = input_detail.get('quantization', (0.0, 0))
            if scale and scale > 0:
                input_data = (input_data / scale + zero_point).astype(np.uint8)
            else:
                input_data = ((input_data + 1.0) * 127.5).astype(np.uint8)
        else:
            input_data = input_data.astype(np.float32)
        
        # Run inference
        with _tflite_lock:
            try:
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                
                embedding = self.interpreter.get_tensor(self.output_details[0]['index'])
            except Exception as e:
                raise RuntimeError(f"Inference failed: {e}")

        output_detail = self.output_details[0]
        if output_detail['dtype'] == np.uint8:
            out_scale, out_zero = output_detail.get('quantization', (0.0, 0))
            if out_scale and out_scale > 0:
                embedding = (embedding.astype(np.float32) - out_zero) * out_scale
            else:
                embedding = embedding.astype(np.float32)
        
        # Output shape typically: (1, embedding_dim)
        embedding = embedding.flatten()
        
        # L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def batch_extract(self, face_images):
        """
        Extract embeddings dari multiple faces
        
        Args:
            face_images (list): List of face images
            
        Returns:
            list: List of embeddings
        """
        embeddings = []
        for face_img in face_images:
            embedding = self.extract_embedding(face_img)
            embeddings.append(embedding)
        return embeddings
    
    # Legacy compatibility methods
    def extract(self, face_image):
        """Legacy method for backward compatibility"""
        return self.extract_embedding(face_image)
    
    @staticmethod
    def _l2_normalize(vector):
        """L2 normalize vector"""
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        """
        Calculate cosine similarity antara dua embedding
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Embedding sudah di-normalize, jadi dot product adalah cosine similarity
        similarity = np.dot(embedding1, embedding2)
        # Clamp ke [-1, 1]
        similarity = np.clip(similarity, -1.0, 1.0)
        # Convert dari [-1, 1] ke [0, 1]
        similarity = (similarity + 1) / 2
        return float(similarity)
    
    @staticmethod
    def euclidean_distance(embedding1, embedding2):
        """
        Calculate Euclidean distance antara dua embedding
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Distance (higher = more different)
        """
        return float(np.linalg.norm(embedding1 - embedding2))


def cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity antara dua embeddings
    
    Args:
        embedding1 (np.ndarray): First embedding
        embedding2 (np.ndarray): Second embedding
        
    Returns:
        float: Similarity score antara 0 dan 1
    """
    # L2 normalized embeddings
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Cosine similarity
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    # Clamp to [0, 1]
    similarity = max(0.0, min(1.0, similarity))
    
    return float(similarity)


class EmbeddingProcessor:
    """Processor untuk mengelola embeddings"""
    
    @staticmethod
    def average_embeddings(embeddings):
        """
        Average multiple embeddings
        
        Args:
            embeddings (list): List of embeddings
            
        Returns:
            np.ndarray: Average embedding (normalized)
        """
        if len(embeddings) == 0:
            raise ValueError("No embeddings provided")
        
        embeddings_array = np.array(embeddings)
        avg_embedding = np.mean(embeddings_array, axis=0)
        
        # L2 normalization
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        return avg_embedding
    
    @staticmethod
    def compare_embeddings(embedding1, embedding2, threshold=0.7):
        """
        Compare dua embeddings dan check apakah match
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
            threshold (float): Similarity threshold untuk match
            
        Returns:
            dict: {
                'similarity': float,
                'is_match': bool
            }
        """
        similarity = cosine_similarity(embedding1, embedding2)
        is_match = similarity >= threshold
        
        return {
            'similarity': similarity,
            'is_match': is_match
        }
