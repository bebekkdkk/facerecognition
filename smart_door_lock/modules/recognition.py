"""
Face Recognition Module
- Compare face embeddings dengan database
- Cosine similarity matching
- Optimized untuk Raspberry Pi 3
"""

import numpy as np
from datetime import datetime
from modules.embedder import cosine_similarity, EmbeddingProcessor


class FaceRecognition:
    """Face recognition engine"""
    
    # Similarity threshold untuk match
    MATCH_THRESHOLD = 0.7
    
    def __init__(self, database=None):
        """
        Initialize face recognition
        
        Args:
            database: FaceDatabase instance
        """
        self.database = database
        self.match_threshold = self.MATCH_THRESHOLD
    
    def compare_with_database(self, embedding, top_k=1):
        """
        Compare embedding dengan semua user embedding di database
        
        Args:
            embedding (np.ndarray): Face embedding dari capture
            top_k (int): Return top-k matches
            
        Returns:
            list: Top-k matches dengan format:
                [{
                    'user_id': str,
                    'username': str,
                    'similarity': float,
                    'is_match': bool
                }]
        """
        if self.database is None:
            return []
        
        try:
            # Get all users dari database
            users = self.database.get_all_users()
            
            if not users:
                return []
            
            results = []
            
            for user in users:
                user_id = user.get('id')
                username = user.get('name', 'Unknown')
                stored_embedding = user.get('embedding')
                
                if stored_embedding is None:
                    continue
                
                # Convert to numpy array jika diperlukan
                if isinstance(stored_embedding, list):
                    stored_embedding = np.array(stored_embedding, dtype=np.float32)
                
                # Ensure same dimension
                if len(embedding) != len(stored_embedding):
                    continue
                
                # Calculate similarity
                similarity = cosine_similarity(embedding, stored_embedding)
                
                # Check if match
                is_match = similarity >= self.match_threshold
                
                results.append({
                    'user_id': user_id,
                    'username': username,
                    'similarity': similarity,
                    'is_match': is_match
                })
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Return top-k
            return results[:top_k]
            
        except Exception as e:
            print(f"[ERROR] Database comparison failed: {e}")
            return []
    
    def recognize(self, embedding, anti_spoof_result=None):
        """
        Recognize face dengan anti-spoofing check
        
        Args:
            embedding (np.ndarray): Face embedding
            anti_spoof_result (dict): Anti-spoofing result
            
        Returns:
            dict: Recognition result
        """
        result = {
            'is_real': True,
            'spoofing_status': 'REAL',
            'recognition_result': None,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check anti-spoofing jika provided
        if anti_spoof_result:
            result['spoofing_status'] = anti_spoof_result.get('label', 'UNKNOWN')
            result['is_real'] = anti_spoof_result.get('is_real', True)
            result['spoofing_confidence'] = anti_spoof_result.get('confidence', 0.0)
            result['laplacian_variance'] = anti_spoof_result.get('laplacian_variance', 0.0)
            
            # Jika FAKE, jangan lanjut ke recognition
            if not result['is_real']:
                result['recognition_result'] = {
                    'matched': False,
                    'username': 'SPOOFING_DETECTED',
                    'similarity': 0.0
                }
                return result
        
        # Jika REAL, jalankan recognition
        matches = self.compare_with_database(embedding, top_k=1)
        
        if matches:
            top_match = matches[0]
            result['recognition_result'] = {
                'matched': top_match['is_match'],
                'username': top_match['username'],
                'user_id': top_match['user_id'],
                'similarity': float(top_match['similarity'])
            }
        else:
            result['recognition_result'] = {
                'matched': False,
                'username': 'UNKNOWN',
                'similarity': 0.0
            }
        
        return result
    
    def get_best_match(self, embedding):
        """
        Get best matching user
        
        Args:
            embedding (np.ndarray): Face embedding
            
        Returns:
            dict or None: Best match atau None jika tidak ada
        """
        matches = self.compare_with_database(embedding, top_k=1)
        
        if matches and matches[0]['is_match']:
            return matches[0]
        
        return None


class RecognitionPipeline:
    """
    Complete recognition pipeline
    - Anti-spoofing + Face recognition
    """
    
    def __init__(self, face_embedder, anti_spoof_detector, database=None):
        """
        Initialize recognition pipeline
        
        Args:
            face_embedder: FaceEmbedder instance
            anti_spoof_detector: FaceAntiSpoofing instance
            database: FaceDatabase instance
        """
        self.face_embedder = face_embedder
        self.anti_spoof_detector = anti_spoof_detector
        self.recognizer = FaceRecognition(database=database)
    
    def process(self, face_image):
        """
        Process face image through complete pipeline
        
        Args:
            face_image (np.ndarray): Cropped face image
            
        Returns:
            dict: Complete result dengan anti-spoofing + recognition
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': None
        }
        
        try:
            # Step 1: Anti-spoofing check
            anti_spoof_result = self.anti_spoof_detector.detect(face_image)
            result['anti_spoof'] = anti_spoof_result
            
            # Jika fake, return langsung
            if not anti_spoof_result.get('is_real', False):
                result['recognition'] = {
                    'matched': False,
                    'username': 'SPOOFING_DETECTED',
                    'similarity': 0.0
                }
                result['success'] = True
                return result
            
            # Step 2: Extract embedding
            embedding = self.face_embedder.extract_embedding(face_image)
            result['embedding_extracted'] = True
            
            # Step 3: Recognition
            recognition_result = self.recognizer.recognize(embedding, anti_spoof_result)
            result['recognition'] = recognition_result.get('recognition_result')
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
            print(f"[ERROR] Pipeline processing failed: {e}")
        
        return result
    
    def batch_process(self, face_images):
        """
        Process multiple face images
        
        Args:
            face_images (list): List of face images
            
        Returns:
            list: List of results
        """
        results = []
        for face_img in face_images:
            result = self.process(face_img)
            results.append(result)
        return results
