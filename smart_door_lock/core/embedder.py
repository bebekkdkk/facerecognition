"""
Face Embedding Extraction menggunakan MobileFaceNet TFLite
"""
import numpy as np
import cv2
import sys
import os
from scipy.spatial.distance import cosine

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
from config import EMBEDDING_MODEL, EMBEDDING


class FaceEmbedder:
    """
    Extract 128-dim face embedding menggunakan MobileFaceNet
    Model: MobileFaceNet.tflite
    """
    
    def __init__(self, model_path=EMBEDDING_MODEL):
        """
        Inisialisasi embedder model
        
        Args:
            model_path: Path ke MobileFaceNet.tflite
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Embedding model not found: {model_path}")
        
        if tflite is None:
            raise RuntimeError("TFLite runtime not available. Install tflite-runtime.")
        
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_size = EMBEDDING['INPUT_SIZE']  # 112
        self.output_size = EMBEDDING['OUTPUT_SIZE']  # 128
        self.norm_mean = EMBEDDING['NORMALIZATION_MEAN']  # 0.5
        self.norm_std = EMBEDDING['NORMALIZATION_STD']  # 0.5
    
    def preprocess_image(self, face_img):
        """
        Preprocess gambar untuk embedding extraction
        
        Flow:
        1. Resize ke 112x112
        2. Convert BGR to RGB
        3. Normalize: (pixel - 0.5) / 0.5
        
        Args:
            face_img: Input face image (BGR)
            
        Returns:
            Preprocessed image array shape (1, 112, 112, 3)
        """
        # 1. Resize ke 112x112
        resized = cv2.resize(face_img, (self.input_size, self.input_size))
        
        # 2. Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 3. Normalize: (pixel - 0.5) / 0.5
        normalized = (rgb.astype(np.float32) - (self.norm_mean * 255)) / (self.norm_std * 255)
        
        # Expand dims untuk batch
        if len(normalized.shape) == 3:
            normalized = np.expand_dims(normalized, axis=0)
        
        return normalized
    
    def l2_normalize(self, vector):
        """
        Apply L2 normalization pada embedding
        
        L2 norm: ||v|| = sqrt(sum(v_i^2))
        Normalized: v / ||v||
        
        Args:
            vector: Embedding vector (128-dim)
            
        Returns:
            L2-normalized vector
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def extract_embedding(self, face_img):
        """
        Extract 128-dim embedding dari face image
        
        Returns:
            128-dim L2-normalized embedding array atau None jika error
        """
        try:
            if face_img is None or face_img.size == 0:
                return None
            
            # Preprocess
            processed = self.preprocess_image(face_img)
            
            # Run TFLite inference
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                processed.astype(self.input_details[0]['dtype'])
            )
            self.interpreter.invoke()
            
            # Get embedding output
            embedding = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            
            # Flatten ke 1D
            embedding = embedding.flatten()
            
            # L2 normalize
            embedding = self.l2_normalize(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None


class EmbeddingProcessor:
    """Process dan manage embeddings"""
    
    @staticmethod
    def average_embeddings(embeddings_list):
        """
        Average multiple embeddings menjadi satu representative embedding
        
        Strategi: Mean vector
        1. Rata-rata semua embeddings
        2. L2 normalize hasil average
        
        Args:
            embeddings_list: List of (128-dim) embeddings
            
        Returns:
            128-dim averaged embedding (L2-normalized) atau None
        """
        if not embeddings_list or len(embeddings_list) == 0:
            return None
        
        embeddings_array = np.array(embeddings_list)
        
        # Mean vector
        avg_embedding = np.mean(embeddings_array, axis=0)
        
        # L2 normalize
        norm = np.linalg.norm(avg_embedding)
        if norm == 0:
            return avg_embedding
        return avg_embedding / norm
    
    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        """
        Calculate cosine similarity antara 2 embeddings
        
        Cosine similarity = 1 - cosine_distance
        Range: 0.0 - 1.0 (higher = more similar)
        
        Args:
            embedding1: First embedding (128-dim)
            embedding2: Second embedding (128-dim)
            
        Returns:
            Similarity score (0.0 - 1.0)
        """
        try:
            if embedding1 is None or embedding2 is None:
                return 0.0
            
            # Cosine similarity = 1 - cosine_distance
            distance = cosine(embedding1, embedding2)
            similarity = 1.0 - distance
            
            # Clamp to [0, 1]
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    @staticmethod
    def filter_embeddings(embeddings_list, similarity_threshold=0.5):
        """
        Filter embeddings berdasarkan similarity dengan mean embedding
        
        Hapus outliers yang terlalu berbeda dari rata-rata
        
        Args:
            embeddings_list: List of embeddings
            similarity_threshold: Minimum similarity to keep
            
        Returns:
            Filtered embeddings_list
        """
        if len(embeddings_list) < 2:
            return embeddings_list
        
        # Calculate mean
        mean_emb = EmbeddingProcessor.average_embeddings(embeddings_list)
        
        # Filter
        filtered = []
        for emb in embeddings_list:
            sim = EmbeddingProcessor.cosine_similarity(emb, mean_emb)
            if sim >= similarity_threshold:
                filtered.append(emb)
        
        return filtered if len(filtered) > 0 else embeddings_list
