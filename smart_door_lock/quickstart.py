"""
Quick Start & Testing Script
- Test semua komponen sistem
- Verify configuration
- Prepare untuk production
"""

import runtime_compat  # noqa: F401 - apply runtime env guards early

import sys
import os

# Python 3.9+ memiliki UTF-8 support default
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    BASE_DIR, DATA_DIR, MODELS_DIR,
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    SIMILARITY_THRESHOLD, EMBEDDING_DIM,
    MOBILEFACENET_PATH,
    HAAR_CASCADE_PATH
)

try:
    import cv2
    import numpy as np
    from modules import (
        ImagePreprocessor, FaceDetector, FaceEmbedder,
        FaceDatabase, FaceTracker
    )
    IMPORTS_OK = True
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    IMPORTS_OK = False


def test_config():
    """Test configuration"""
    print("\n" + "="*40)
    print("CONFIGURATION TEST")
    print("="*40 + "\n")
    
    print(f"Base Directory:      {BASE_DIR}")
    print(f"Data Directory:      {DATA_DIR}")
    print(f"Models Directory:    {MODELS_DIR}")
    print(f"Camera Index:        {CAMERA_INDEX}")
    print(f"Frame Size:          {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Embedding Dim:       {EMBEDDING_DIM}")
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD}")
    
    # Check directories
    print("\n[CHECKING] Directories...")
    for dir_path in [DATA_DIR, MODELS_DIR]:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path}")
    
    print("\n[SUCCESS] Configuration loaded!")


def test_camera():
    """Test camera access"""
    print("\n╔════════════════════════════════════════╗")
    print("║         CAMERA TEST                    ║")
    print("╚════════════════════════════════════════╝\n")
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"[ERROR] Camera {CAMERA_INDEX} not accessible!")
        return False
    
    ret, frame = cap.read()
    if ret:
        print(f"[SUCCESS] Camera opened successfully!")
        print(f"  - Resolution: {frame.shape[1]}x{frame.shape[0]}")
        
        # Take 5 frames untuk test
        for i in range(5):
            ret, _ = cap.read()
            if ret:
                print(f"  - Frame {i+1}: OK")
            else:
                print(f"  - Frame {i+1}: FAILED")
                return False
        
        cap.release()
        return True
    else:
        print("[ERROR] Failed to capture frame!")
        cap.release()
        return False


def test_modules():
    """Test system modules"""
    print("\n╔════════════════════════════════════════╗")
    print("║        MODULES TEST                    ║")
    print("╚════════════════════════════════════════╝\n")
    
    if not IMPORTS_OK:
        print("[ERROR] Imports failed!")
        return False
    
    try:
        print("[TESTING] ImagePreprocessor...")
        preprocessor = ImagePreprocessor()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        resized = preprocessor.resize_frame(frame)
        print(f"  ✓ ImagePreprocessor OK (output shape: {resized.shape})")
        
        print("\n[TESTING] FaceDetector...")
        detector = FaceDetector(HAAR_CASCADE_PATH)
        detections = detector.detect(frame)
        print(f"  ✓ FaceDetector OK (found {len(detections)} faces)")
        
        print("\n[TESTING] FaceEmbedder...")
        embedder = FaceEmbedder(MOBILEFACENET_PATH)
        test_face = np.zeros((112, 112, 3), dtype=np.uint8)
        embedding = embedder.extract_embedding(test_face)
        print(f"  ✓ FaceEmbedder OK (embedding shape: {embedding.shape})")
        
        print("\n[TESTING] FaceDatabase...")
        database = FaceDatabase()
        database.create_table()
        stats = database.get_stats()
        print(f"  ✓ FaceDatabase OK")
        print(f"    - Total embeddings: {stats.get('total_embeddings', 0)}")
        print(f"    - Total users: {stats.get('total_users', 0)}")
        
        print("\n[TESTING] FaceTracker...")
        tracker = FaceTracker()
        test_detections = [(100, 100, 50, 50, 0.9)]
        tracked = tracker.update(test_detections)
        print(f"  ✓ FaceTracker OK (tracked {len(tracked)} objects)")
        
        print("\n[SUCCESS] All modules loaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_similarity():
    """Test embedding similarity"""
    print("\n╔════════════════════════════════════════╗")
    print("║       SIMILARITY TEST                  ║")
    print("╚════════════════════════════════════════╝\n")
    
    from modules.embedder import cosine_similarity
    
    # Create test embeddings
    emb1 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    emb1_norm = np.linalg.norm(emb1)
    if emb1_norm > 0:
        emb1 = emb1 / emb1_norm
    
    emb2 = emb1 + np.random.randn(EMBEDDING_DIM).astype(np.float32) * 0.01  # Similar
    emb2_norm = np.linalg.norm(emb2)
    if emb2_norm > 0:
        emb2 = emb2 / emb2_norm
    
    emb3 = np.random.randn(EMBEDDING_DIM).astype(np.float32)  # Different
    emb3_norm = np.linalg.norm(emb3)
    if emb3_norm > 0:
        emb3 = emb3 / emb3_norm
    
    # Test similarity
    sim1_2 = cosine_similarity(emb1, emb2)
    sim1_3 = cosine_similarity(emb1, emb3)
    
    print(f"Similarity (same): {sim1_2:.4f}")
    print(f"Similarity (diff): {sim1_3:.4f}")
    print(f"Threshold:         {SIMILARITY_THRESHOLD:.4f}")
    
    if sim1_2 > SIMILARITY_THRESHOLD:
        print("\n[SUCCESS] Similar embedding correctly identified!")
    else:
        print("\n[WARNING] Adjust SIMILARITY_THRESHOLD in config.py")
    
    if sim1_3 < SIMILARITY_THRESHOLD:
        print("[SUCCESS] Different embedding correctly rejected!")
    else:
        print("[WARNING] Threshold may be too low!")


def print_next_steps():
    """Print next steps"""
    print("\n╔════════════════════════════════════════╗")
    print("║         NEXT STEPS                     ║")
    print("╚════════════════════════════════════════╝\n")
    
    print("1. ENROLLMENT - Register users:")
    print("   $ python enrollment.py\n")
    
    print("2. AUTHENTICATION - Test access control:")
    print("   $ python authenticate.py\n")
    
    print("3. ADMIN - Manage database:")
    print("   $ python admin.py\n")
    
    print("For more information, see README.md")


def main():
    """Main test suite"""
    
    print("\n" + "="*50)
    print("SMART DOOR LOCK - SYSTEM TEST")
    print("="*50)
    
    # Run tests
    test_config()
    
    camera_ok = test_camera()
    if not camera_ok:
        print("\n[WARNING] Camera test failed!")
        print("Check CAMERA_INDEX in config.py")
    
    modules_ok = test_modules()
    if modules_ok:
        test_similarity()
        print_next_steps()
    else:
        print("\n[ERROR] Please fix module errors first!")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("[SUMMARY] System ready for deployment!")
    print("="*50 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[ABORT] Test interrupted!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
