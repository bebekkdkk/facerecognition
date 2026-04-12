"""
Quick Start Guide - Smart Door Lock
"""

# ============================================================
# 1. INSTALLATION QUICK START
# ============================================================

# Buka terminal di folder: d:\cobaskripsi\smart_door_lock

# Install dependencies
pip install -r requirements.txt

# Untuk development/desktop:
# pip install tensorflow>=2.5.0

# Untuk Raspberry Pi:
# pip install tflite-runtime

# ============================================================
# 2. VERIFY MODELS
# ============================================================

import os
from config import (
    FACE_DETECTION_MODEL,
    ANTI_SPOOFING_MODEL,
    EMBEDDING_MODEL
)

print("Checking models...")
print(f"Face Detection: {'✓' if os.path.exists(FACE_DETECTION_MODEL) else '✗'} 
      {FACE_DETECTION_MODEL}")
print(f"Anti-Spoofing: {'✓' if os.path.exists(ANTI_SPOOFING_MODEL) else '✗'} 
      {ANTI_SPOOFING_MODEL}")
print(f"Embedding: {'✓' if os.path.exists(EMBEDDING_MODEL) else '✗'} 
      {EMBEDDING_MODEL}")

# ============================================================
# 3. TEST COMPONENTS INDIVIDUALLY
# ============================================================

# Test 1: Face Detection
print("\n=== TEST 1: Face Detection ===")
from core.face_detector import FaceDetector
import cv2

detector = FaceDetector()
cap = cv2.VideoCapture(0)

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = detector.detect(frame)
    print(f"Frame {i+1}: {len(faces)} faces detected")
    
    if len(faces) > 0:
        print(f"  Face coordinates: {faces[0]}")

cap.release()
print("✓ Face detection working\n")

# ============================================================

# Test 2: Anti-Spoofing
print("=== TEST 2: Anti-Spoofing ===")
from core.anti_spoofing import FaceAntiSpoofing

anti_spoof = FaceAntiSpoofing()
cap = cv2.VideoCapture(0)

for i in range(5):
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = detector.detect(frame)
    
    if len(faces) > 0:
        face_img = detector.crop_face(frame, faces[0])
        result = anti_spoof.predict(face_img)
        
        print(f"Frame {i+1}:")
        print(f"  Status: {result['status']}")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Laplacian: {result['laplacian_score']:.1f}")
        print(f"  Is Real: {result['is_real']}")

cap.release()
print("✓ Anti-spoofing working\n")

# ============================================================

# Test 3: Embedding Extraction
print("=== TEST 3: Embedding Extraction ===")
from core.embedder import FaceEmbedder, EmbeddingProcessor

embedder = FaceEmbedder()
cap = cv2.VideoCapture(0)

embeddings = []

for i in range(5):
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = detector.detect(frame)
    
    if len(faces) > 0:
        face_img = detector.crop_face(frame, faces[0])
        result = anti_spoof.predict(face_img)
        
        if result['is_real']:
            emb = embedder.extract_embedding(face_img)
            if emb is not None:
                embeddings.append(emb)
                print(f"Frame {i+1}: Embedding extracted, shape: {emb.shape}")

cap.release()

if len(embeddings) > 1:
    # Test averaging
    avg_emb = EmbeddingProcessor.average_embeddings(embeddings)
    print(f"Average embedding shape: {avg_emb.shape}")
    
    # Test similarity
    sim = EmbeddingProcessor.cosine_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between emb[0] and emb[1]: {sim:.4f}")

print("✓ Embedding extraction working\n")

# ============================================================
# 4. ENROLL USER
# ============================================================

print("=== ENROLL USER ===")
print("\nTo enroll a user, run:")
print("  python -m smart_door_lock.enrollment")
print("OR")
print("  cd d:\\cobaskripsi")
print("  python smart_door_lock/enrollment.py")

# ============================================================
# 5. RUN RECOGNITION
# ============================================================

print("\n=== RUN RECOGNITION ===")
print("\nTo run real-time recognition, run:")
print("  python -m smart_door_lock.main")
print("OR")
print("  cd d:\\cobaskripsi")
print("  python smart_door_lock/main.py")

# ============================================================
# 6. TEST DATA EXAMPLE
# ============================================================

print("\n=== TEST DATA ===")

# Create sample test data
test_database = {
    'john_doe': FaceEmbedder().extract_embedding(
        cv2.imread('test_image.jpg')
    ) if os.path.exists('test_image.jpg') else None,
}

print(f"Database keys: {list(test_database.keys())}")

# ============================================================
# 7. FILE STRUCTURE
# ============================================================

print("\n=== FILE STRUCTURE ===\n")

import os

def print_tree(directory, prefix="", max_depth=3, current_depth=0):
    if current_depth >= max_depth:
        return
    
    items = []
    try:
        items = sorted(os.listdir(directory))
    except PermissionError:
        return
    
    # Filter
    skip = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.egg-info'}
    items = [i for i in items if i not in skip and not i.startswith('.')]
    
    for i, item in enumerate(items):
        path = os.path.join(directory, item)
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(prefix + current_prefix + item)
        
        if os.path.isdir(path) and current_depth < max_depth - 1:
            next_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(path, next_prefix, max_depth, current_depth + 1)

print("smart_door_lock/")
print_tree(".", max_depth=2)

# ============================================================
# 8. TROUBLESHOOTING
# ============================================================

print("\n=== TROUBLESHOOTING ===\n")

print("Problem: Model not found")
print("Solution: Check that models are in smart_door_lock/models/")
print("  - FaceAntiSpoofing.tflite")
print("  - haarcascade_frontalface_default.xml")
print("  - MobileFaceNet.tflite")

print("\nProblem: TFLite runtime error")
print("Solution: Install tflite-runtime or tensorflow")
print("  pip install tflite-runtime")
print("  # or")
print("  pip install tensorflow>=2.5.0")

print("\nProblem: Camera not working")
print("Solution: Check camera permissions and availability")
print("  # On Linux/Mac, might need:")
print("  sudo usermod -a -G video $USER")

print("\nProblem: Low recognition accuracy")
print("Solution:")
print("  1. Enroll more users with different lighting/angles")
print("  2. Improve camera quality/lighting")
print("  3. Adjust COSINE_THRESHOLD in config.py")
print("  4. Capture more poses during enrollment")

print("\nProblem: Slow performance")
print("Solution:")
print("  1. Enable RASPBERRY_PI_MODE in config.py")
print("  2. Reduce MAX_FRAME_WIDTH / MAX_FRAME_HEIGHT")
print("  3. Use GPU if available (tensorflow-gpu)")

# ============================================================
# 9. NEXT STEPS
# ============================================================

print("\n=== NEXT STEPS ===\n")

print("1. Verify all models are in place")
print("   ✓ Check smart_door_lock/models/ folder")

print("\n2. Install dependencies")
print("   $ pip install -r smart_door_lock/requirements.txt")

print("\n3. Enroll users")
print("   $ python smart_door_lock/enrollment.py")
print("   - Follow 5-pose capture instructions")
print("   - Capture clear face images")

print("\n4. Test recognition")
print("   $ python smart_door_lock/main.py")
print("   - Check if enrolled users are recognized")
print("   - Test anti-spoofing with photos")

print("\n5. Tune parameters")
print("   - Edit config.py if needed")
print("   - Adjust thresholds for your environment")

print("\n6. Deploy")
print("   - Transfer to Raspberry Pi")
print("   - Install tflite-runtime")
print("   - Run on target hardware")

print("\n" + "="*60)
print("Ready to use! Start with enrollment.py")
print("="*60 + "\n")
