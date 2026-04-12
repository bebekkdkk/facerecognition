"""
Unit Testing - Smart Door Lock Components
Test setiap modul secara individual
"""

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import time
from core.face_detector import FaceDetector
from core.anti_spoofing import FaceAntiSpoofing
from core.embedder import FaceEmbedder, EmbeddingProcessor
from core.recognition import FaceRecognition, RecognitionPipeline
from config import (
    FACE_DETECTION_MODEL,
    ANTI_SPOOFING_MODEL,
    EMBEDDING_MODEL,
)


class TestRunner:
    """Test runner untuk Smart Door Lock components"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests_run = 0
    
    def test(self, name, condition, expected=True):
        """Run a test"""
        self.tests_run += 1
        status = "✓" if condition == expected else "✗"
        
        if condition == expected:
            self.passed += 1
            print(f"  {status} {name}")
        else:
            self.failed += 1
            print(f"  {status} {name} - FAILED")
            print(f"    Expected: {expected}, Got: {condition}")
        
        return condition == expected
    
    def report(self):
        """Print test report"""
        total = self.passed + self.failed
        percentage = (self.passed / total * 100) if total > 0 else 0
        
        print("\n" + "="*60)
        print("TEST REPORT")
        print("="*60)
        print(f"Total tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Pass rate: {percentage:.1f}%")
        print("="*60 + "\n")
        
        return self.failed == 0


def test_face_detector(runner):
    """Test FaceDetector component"""
    print("\n=== Testing FaceDetector ===")
    
    try:
        detector = FaceDetector()
        runner.test("FaceDetector initialization", True)
    except Exception as e:
        runner.test("FaceDetector initialization", False)
        print(f"  Error: {e}")
        return
    
    # Test with dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    try:
        faces = detector.detect(dummy_frame)
        runner.test("Face detection on dummy frame", isinstance(faces, np.ndarray))
        runner.test("Face detection returns empty for dummy", len(faces) == 0)
    except Exception as e:
        runner.test("Face detection", False)
        print(f"  Error: {e}")
    
    # Test crop_face
    try:
        if len(faces) == 0:
            # Create a valid face coordinate
            face_coords = (100, 100, 50, 50)
            cropped = detector.crop_face(dummy_frame, face_coords)
            runner.test("Crop face returns correct shape", 
                       cropped.shape == (50, 50, 3))
    except Exception as e:
        runner.test("Crop face", False)
        print(f"  Error: {e}")
    
    # Test resize_face
    try:
        face_img = np.zeros((50, 50, 3), dtype=np.uint8)
        resized = detector.resize_face(face_img, (112, 112))
        runner.test("Resize face returns correct size", 
                   resized.shape == (112, 112, 3))
    except Exception as e:
        runner.test("Resize face", False)
        print(f"  Error: {e}")


def test_anti_spoofing(runner):
    """Test FaceAntiSpoofing component"""
    print("\n=== Testing FaceAntiSpoofing ===")
    
    try:
        anti_spoof = FaceAntiSpoofing()
        runner.test("FaceAntiSpoofing initialization", True)
    except Exception as e:
        runner.test("FaceAntiSpoofing initialization", False)
        print(f"  Error: {e}")
        return
    
    # Test with dummy face image
    dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    try:
        result = anti_spoof.predict(dummy_face)
        runner.test("Anti-spoofing prediction returns dict", isinstance(result, dict))
        runner.test("Anti-spoofing result has 'is_real' key", 'is_real' in result)
        runner.test("Anti-spoofing result has 'score' key", 'score' in result)
        runner.test("Anti-spoofing result has 'status' key", 'status' in result)
        runner.test("Anti-spoofing status is valid", 
                   result['status'] in ['REAL', 'FAKE', 'BLURRED', 'ERROR', 'UNKNOWN'])
    except Exception as e:
        runner.test("Anti-spoofing prediction", False)
        print(f"  Error: {e}")
    
    # Test laplacian
    try:
        gray_face = cv2.cvtColor(dummy_face, cv2.COLOR_BGR2GRAY)
        laplacian_score = anti_spoof.calculate_laplacian(gray_face)
        runner.test("Laplacian calculation returns float", isinstance(laplacian_score, (float, np.floating)))
        runner.test("Laplacian score is positive", laplacian_score >= 0)
    except Exception as e:
        runner.test("Laplacian calculation", False)
        print(f"  Error: {e}")


def test_embedder(runner):
    """Test FaceEmbedder component"""
    print("\n=== Testing FaceEmbedder ===")
    
    try:
        embedder = FaceEmbedder()
        runner.test("FaceEmbedder initialization", True)
    except Exception as e:
        runner.test("FaceEmbedder initialization", False)
        print(f"  Error: {e}")
        return
    
    # Test with dummy face image
    dummy_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    try:
        embedding = embedder.extract_embedding(dummy_face)
        runner.test("Embedding extraction returns ndarray", isinstance(embedding, np.ndarray))
        runner.test("Embedding has correct shape", embedding.shape == (128,))
        runner.test("Embedding is normalized", np.abs(np.linalg.norm(embedding) - 1.0) < 0.01)
    except Exception as e:
        runner.test("Embedding extraction", False)
        print(f"  Error: {e}")
        return
    
    # Test L2 normalization
    try:
        test_vec = np.array([3.0, 4.0])
        normalized = embedder.l2_normalize(test_vec)
        norm_length = np.linalg.norm(normalized)
        runner.test("L2 normalization produces unit vector", 
                   np.abs(norm_length - 1.0) < 0.01)
    except Exception as e:
        runner.test("L2 normalization", False)
        print(f"  Error: {e}")


def test_embedding_processor(runner):
    """Test EmbeddingProcessor component"""
    print("\n=== Testing EmbeddingProcessor ===")
    
    # Create test embeddings
    emb1 = np.random.randn(128)
    emb1 = emb1 / np.linalg.norm(emb1)
    
    emb2 = np.random.randn(128)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    emb3 = np.random.randn(128)
    emb3 = emb3 / np.linalg.norm(emb3)
    
    # Test average
    try:
        avg = EmbeddingProcessor.average_embeddings([emb1, emb2, emb3])
        runner.test("Average embeddings returns ndarray", isinstance(avg, np.ndarray))
        runner.test("Average embeddings has correct shape", avg.shape == (128,))
        runner.test("Averaged embedding is normalized", 
                   np.abs(np.linalg.norm(avg) - 1.0) < 0.01)
    except Exception as e:
        runner.test("Average embeddings", False)
        print(f"  Error: {e}")
    
    # Test cosine similarity
    try:
        sim = EmbeddingProcessor.cosine_similarity(emb1, emb1)
        runner.test("Cosine similarity of identical embeddings ~1.0", 
                   np.abs(sim - 1.0) < 0.01)
        
        sim2 = EmbeddingProcessor.cosine_similarity(emb1, emb2)
        runner.test("Cosine similarity is in range [0, 1]", 0.0 <= sim2 <= 1.0)
    except Exception as e:
        runner.test("Cosine similarity", False)
        print(f"  Error: {e}")


def test_face_recognition(runner):
    """Test FaceRecognition component"""
    print("\n=== Testing FaceRecognition ===")
    
    try:
        recognizer = FaceRecognition(threshold=0.7)
        runner.test("FaceRecognition initialization", True)
    except Exception as e:
        runner.test("FaceRecognition initialization", False)
        print(f"  Error: {e}")
        return
    
    # Create test embeddings
    query_emb = np.random.randn(128)
    query_emb = query_emb / np.linalg.norm(query_emb)
    
    database = {
        'user1': query_emb.copy(),  # Will match
        'user2': np.random.randn(128),
        'user3': np.random.randn(128),
    }
    
    for key in database:
        database[key] = database[key] / np.linalg.norm(database[key])
    
    # Test matching
    try:
        result = recognizer.match(query_emb, database)
        runner.test("Match returns dict", isinstance(result, dict))
        runner.test("Match result has 'matched' key", 'matched' in result)
        runner.test("Match result has 'user_id' key", 'user_id' in result)
        runner.test("Match result has 'similarity' key", 'similarity' in result)
        runner.test("Match result has 'top_matches' key", 'top_matches' in result)
        
        # Should match user1
        runner.test("Query matches itself in database", result['matched'] == True)
        runner.test("Matched user is 'user1'", result['user_id'] == 'user1')
        runner.test("Similarity score high", result['similarity'] > 0.9)
    except Exception as e:
        runner.test("Face recognition matching", False)
        print(f"  Error: {e}")


def test_models_exist(runner):
    """Test that all models exist"""
    print("\n=== Testing Model Files ===")
    
    runner.test("Face detection model exists", os.path.exists(FACE_DETECTION_MODEL))
    runner.test("Anti-spoofing model exists", os.path.exists(ANTI_SPOOFING_MODEL))
    runner.test("Embedding model exists", os.path.exists(EMBEDDING_MODEL))


def main():
    """Run all tests"""
    print("="*60)
    print("SMART DOOR LOCK - COMPONENT TESTING")
    print("="*60)
    
    runner = TestRunner()
    
    # Run tests
    test_models_exist(runner)
    test_face_detector(runner)
    test_anti_spoofing(runner)
    test_embedder(runner)
    test_embedding_processor(runner)
    test_face_recognition(runner)
    
    # Report
    success = runner.report()
    
    if success:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
