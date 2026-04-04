"""
Face Embedding Module
- Extract face embeddings menggunakan ArcFace model
- L2 normalization
"""

import cv2
import numpy as np

# Try to import onnxruntime, fallback to CPU-only mode if not available
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None
    print("[WARNING] onnxruntime not installed. Will use feature-based embedding only.")

from config import TARGET_FACE_SIZE, EMBEDDING_DIM


class FaceEmbedder:
    """Face embedder menggunakan ArcFace ONNX model"""
    
    def __init__(self, onnx_path=None):
        """
        Initialize face embedder
        
        Args:
            onnx_path: Path ke ArcFace ONNX model
        """
        self.onnx_path = onnx_path
        self.onnx_session = None
        self.use_onnx = False
        
        if not ONNX_AVAILABLE:
            print("[INFO] ONNX Runtime not available, using fallback embedding")
            return
        
        if onnx_path:
            try:
                self.onnx_session = ort.InferenceSession(
                    onnx_path,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                self.use_onnx = True
                print(f"[INFO] ArcFace ONNX model loaded: {onnx_path}")
            except Exception as e:
                print(f"[WARNING] ArcFace ONNX model loading failed: {e}")
                print("[INFO] Will use feature-based embedding")
    
    def simple_embedding(self, face_img):
        """
        Simple feature-based embedding sebagai fallback
        Menggunakan histogram dan edge features
        
        Args:
            face_img: Preprocessed face image (112x112)
            
        Returns:
            Embedding vector (512,)
        """
        # Ensure correct size
        if face_img.shape != TARGET_FACE_SIZE:
            face_img = cv2.resize(face_img, TARGET_FACE_SIZE)
        
        # Convert to grayscale jika kondisi tertentu
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Ensure uint8 untuk histogram calculation
        if gray.dtype != np.uint8:
            if gray.max() <= 1:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
        
        # Extract histogram features
        hist_features = []
        for i in range(0, 112, 28):  # 4x4 grid
            for j in range(0, 112, 28):
                patch = gray[i:i+28, j:j+28]
                hist = cv2.calcHist([patch], [0], None, [32], [0, 256])
                hist_features.extend(hist.flatten())
        
        # Extract edge features
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
        
        # Combine features - convert semua ke numpy array dulu
        hist_array = np.array(hist_features).flatten()
        edge_array = edge_hist.flatten()
        
        # Pad edge histogram ke same size dengan hist
        if len(edge_array) < len(hist_array):
            edge_array = np.pad(edge_array, (0, len(hist_array) - len(edge_array)))
        
        # Concatenate properly
        embedding = np.concatenate([hist_array, edge_array[:len(hist_array)]])
        
        # Pad atau trim ke 512
        if len(embedding) < EMBEDDING_DIM:
            embedding = np.pad(embedding, (0, EMBEDDING_DIM - len(embedding)))
        else:
            embedding = embedding[:EMBEDDING_DIM]
        
        # L2 normalize
        embedding = self._l2_normalize(embedding)
        
        return embedding.astype(np.float32)
    
    def extract_with_onnx(self, face_img):
        """
        Extract embedding menggunakan ArcFace ONNX
        
        Args:
            face_img: Preprocessed face image (112x112, float32, [0,1])
            
        Returns:
            Embedding vector (512,)
        """
        if self.onnx_session is None:
            return self.simple_embedding(face_img)
        
        try:
            # Ensure correct shape dan type
            if face_img.shape != TARGET_FACE_SIZE:
                face_img = cv2.resize(face_img, TARGET_FACE_SIZE)
            
            # Konversi ke float32 jika belum
            if face_img.dtype != np.float32:
                face_img = face_img.astype(np.float32)
            
            # Prepare input
            if len(face_img.shape) == 2:  # Grayscale
                face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
            
            # NCHW format
            img_input = face_img.transpose(2, 0, 1)[np.newaxis, ...]
            
            # Inference
            input_name = self.onnx_session.get_inputs()[0].name
            embedding = self.onnx_session.run(None, {input_name: img_input})
            
            # Output adalah embedding
            embedding = embedding[0][0]  # Take first batch
            
            # L2 normalize
            embedding = self._l2_normalize(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            print(f"[ERROR] ONNX inference failed: {e}")
            return self.simple_embedding(face_img)
    
    def extract(self, face_img):
        """
        Extract embedding dari face image
        
        Args:
            face_img: Preprocessed face image
            
        Returns:
            Embedding vector (512,)
        """
        if self.use_onnx:
            return self.extract_with_onnx(face_img)
        else:
            return self.simple_embedding(face_img)
    
    @staticmethod
    def _l2_normalize(vector):
        """
        L2 normalize vector
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
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
        # Clamp ke [0, 1]
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
