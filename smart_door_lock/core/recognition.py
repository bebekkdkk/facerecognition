"""
Face Recognition - Matching embeddings dengan database
"""
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .embedder import EmbeddingProcessor
from config import RECOGNITION


class FaceRecognition:
    """Matching embedding dengan database"""
    
    def __init__(self, threshold=RECOGNITION['COSINE_THRESHOLD']):
        """
        Inisialisasi face recognition
        
        Args:
            threshold: Cosine similarity threshold (default 0.7)
        """
        self.threshold = threshold
    
    def match(self, embedding, database):
        """
        Match embedding dengan database
        
        Flow:
        1. Bandingkan embedding dengan SEMUA user embeddings di database
        2. Hitung cosine similarity untuk setiap user
        3. Sort by similarity (descending)
        4. Check threshold >= 0.7 → MATCH
        
        Args:
            embedding: Query embedding (128-dim)
            database: Dict {user_id: stored_embedding}
            
        Returns:
            Dict dengan keys:
                - matched: bool (True jika similarity >= threshold)
                - user_id: str atau None (user ID jika matched)
                - similarity: float (best similarity score)
                - top_matches: list of (user_id, similarity) tuples (top 5)
        """
        result = {
            'matched': False,
            'user_id': None,
            'similarity': 0.0,
            'top_matches': []
        }
        
        if not database or len(database) == 0:
            return result
        
        if embedding is None:
            return result
        
        similarities = {}
        
        # Bandingkan ke SEMUA user embeddings
        for user_id, stored_embedding in database.items():
            if stored_embedding is None:
                continue
            
            similarity = EmbeddingProcessor.cosine_similarity(
                embedding,
                stored_embedding
            )
            similarities[user_id] = similarity
        
        if not similarities:
            return result
        
        # Sort by similarity (descending)
        sorted_matches = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Store top 5 matches
        result['top_matches'] = sorted_matches[:5]
        
        if sorted_matches:
            best_user, best_similarity = sorted_matches[0]
            result['similarity'] = best_similarity
            
            # Check threshold >= 0.7 → MATCH
            if best_similarity >= self.threshold:
                result['matched'] = True
                result['user_id'] = best_user
        
        return result


class RecognitionPipeline:
    """
    Pipeline lengkap: anti-spoofing + embedding extraction + recognition
    
    Flow:
    1. Detect faces
    2. Anti-spoofing check FIRST
    3. Jika FAKE → reject (no embedding extraction)
    4. Jika REAL → extract embedding dan match dengan database
    """
    
    def __init__(self, face_detector, anti_spoofing, embedder, database, 
                 recognition_threshold=RECOGNITION['COSINE_THRESHOLD']):
        """
        Inisialisasi recognition pipeline
        
        Args:
            face_detector: FaceDetector instance
            anti_spoofing: FaceAntiSpoofing instance
            embedder: FaceEmbedder instance
            database: Dict {user_id: embedding}
            recognition_threshold: Cosine threshold (default 0.7)
        """
        self.face_detector = face_detector
        self.anti_spoofing = anti_spoofing
        self.embedder = embedder
        self.database = database
        self.recognizer = FaceRecognition(threshold=recognition_threshold)
    
    def process_frame(self, frame):
        """
        Process frame lengkap: detect → anti-spoofing → embedding → match
        
        Flow:
        1. Detect wajah
        2. Untuk setiap wajah:
           a. Anti-spoofing check FIRST
           b. Jika FAKE → reject (no embedding extraction)
           c. Jika REAL → extract embedding
           d. Match dengan database
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of result dicts, setiap result:
                - face_coords: (x, y, w, h)
                - face_img: cropped face image
                - spoofing_result: dict dari anti_spoofing
                - is_real: bool
                - embedding: extracted embedding atau None
                - recognition_result: match result atau None
        """
        results = []
        
        # 1. Detect faces
        faces = self.face_detector.detect(frame)
        
        if len(faces) == 0:
            return results
        
        # 2. Process setiap wajah
        for face_coords in faces:
            result = {
                'face_coords': tuple(face_coords),
                'face_img': None,
                'spoofing_result': None,
                'is_real': False,
                'embedding': None,
                'recognition_result': None
            }
            
            # Extract face image
            face_img = self.face_detector.crop_face(frame, face_coords)
            if face_img is None or face_img.size == 0:
                continue
            
            result['face_img'] = face_img
            
            # Anti-spoofing check FIRST
            spoofing_result = self.anti_spoofing.predict(face_img)
            result['spoofing_result'] = spoofing_result
            result['is_real'] = spoofing_result['is_real']
            
            # Jika FAKE → reject
            if not spoofing_result['is_real']:
                results.append(result)
                continue
            
            # Jika REAL → extract embedding
            embedding = self.embedder.extract_embedding(face_img)
            result['embedding'] = embedding
            
            if embedding is not None:
                # Match dengan database
                recognition_result = self.recognizer.match(embedding, self.database)
                result['recognition_result'] = recognition_result
            
            results.append(result)
        
        return results
    
    def update_database(self, user_id, embedding):
        """
        Update database dengan user baru atau update existing
        
        Args:
            user_id: User identifier (string)
            embedding: 128-dim embedding array
        """
        self.database[user_id] = embedding
    
    def remove_user(self, user_id):
        """
        Remove user dari database
        
        Args:
            user_id: User identifier
        """
        if user_id in self.database:
            del self.database[user_id]
    
    def get_database(self):
        """Get copy database"""
        return self.database.copy()
    
    def get_users_count(self):
        """Get jumlah user di database"""
        return len(self.database)
