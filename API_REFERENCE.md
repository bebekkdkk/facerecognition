"""
MODULE API DOCUMENTATION
Smart Door Lock - Face Recognition + Anti-Spoofing System
"""

# ============================================================
# 1. anti_spoofing.py
# ============================================================

"""
FaceAntiSpoofing
- Main class untuk deteksi live vs spoof attack
- Input: cropped face image (BGR dari OpenCV)
- Output: dict dengan status (REAL/FAKE), score, confidence

Key Attributes:
    INPUT_IMAGE_SIZE = 256        # Model input size
    THRESHOLD = 0.2               # Score threshold (< 0.2 = REAL)
    LAPLACE_THRESHOLD = 50        # Blur variance threshold
    LAPLACIAN_THRESHOLD = 1000    # Additional threshold

Methods:
    __init__(model_path)          # Initialize dengan TFLite model
    detect(face_image)            # Process satu face image
    batch_detect(face_images)     # Process multiple faces
    
Returns from detect():
    {
        'is_real': bool,              # True = REAL, False = FAKE
        'score': float,               # Raw model score
        'laplacian_variance': float,  # Blur metric
        'label': str,                 # 'REAL' or 'FAKE'
        'confidence': float,          # 0.0-1.0
        'raw_output': list,           # Raw model output
        'reason': str                 # Rejection reason jika ada
    }

Example Usage:
    detector = FaceAntiSpoofing("models/FaceAntiSpoofing.tflite")
    result = detector.detect(face_image)
    if result['is_real']:
        print(f"Real face detected! Confidence: {result['confidence']}")
    else:
        print("Spoofing detected!")

AntiSpoofingPipeline
- Wrapper yang lebih simple untuk anti-spoofing detection
- Convenience layer atas FaceAntiSpoofing

Methods:
    __init__(model_path)
    process(face_image) → simplified result

"""

# ============================================================
# 2. embedder.py
# ============================================================

"""
FaceEmbedder
- Extract face embeddings menggunakan MobileFaceNet.tflite
- Output: 128-dimensional L2-normalized vector

Key Attributes:
    TARGET_FACE_SIZE = (112, 112)  # Model input size
    EMBEDDING_DIM = 128            # Output dimension

Methods:
    __init__(model_path)
    extract_embedding(face_image)  # Main method
    batch_extract(face_images)     # Process multiple faces
    preprocess_face(face_image)    # Preprocessing step

Returns from extract_embedding():
    np.ndarray (128,)              # L2 normalized embedding

Example Usage:
    embedder = FaceEmbedder("models/MobileFaceNet.tflite")
    embedding = embedder.extract_embedding(face_image)
    print(f"Embedding shape: {embedding.shape}")
    print(f"L2 norm: {np.linalg.norm(embedding)}")  # Should be ~1.0

cosine_similarity(embedding1, embedding2)
- Function untuk hitung cosine similarity antara 2 embeddings
- Input: 2 embedding vectors
- Output: similarity score 0.0-1.0
- Note: embeddings harus L2 normalized

Example Usage:
    sim = cosine_similarity(emb1, emb2)
    print(f"Similarity: {sim:.3f}")  # e.g., 0.875

EmbeddingProcessor
- Utilities untuk manage embeddings
- Static methods only

Methods:
    average_embeddings(embeddings)  # Average multiple embeddings
    compare_embeddings(emb1, emb2, threshold=0.7)  # Compare & match

Example Usage:
    # Average 5 pose embeddings
    avg_emb = EmbeddingProcessor.average_embeddings([emb1, emb2, emb3, emb4, emb5])
    
    # Compare dengan stored embedding
    result = EmbeddingProcessor.compare_embeddings(captured_emb, stored_emb, 0.7)
    if result['is_match']:
        print(f"Match! Similarity: {result['similarity']:.3f}")

"""

# ============================================================
# 3. recognition.py
# ============================================================

"""
FaceRecognition
- Compare embeddings dengan database
- Find best matching user
- Cosine similarity matching

Key Attributes:
    MATCH_THRESHOLD = 0.7  # Similarity threshold

Methods:
    __init__(database)
    compare_with_database(embedding, top_k=1)  # Find top-k matches
    recognize(embedding, anti_spoof_result)    # Full recognition
    get_best_match(embedding)                  # Get single best match

Returns from compare_with_database():
    list of dicts:
    [{
        'user_id': str,
        'username': str,
        'similarity': float,  # 0.0-1.0
        'is_match': bool      # True if >= threshold
    }]

Example Usage:
    recognizer = FaceRecognition(database)
    matches = recognizer.compare_with_database(embedding, top_k=3)
    for match in matches:
        print(f"{match['username']}: {match['similarity']:.3f}")

RecognitionPipeline
- Complete pipeline: anti-spoofing + embedding + recognition
- Main class untuk integration seluruh system

Methods:
    __init__(face_embedder, anti_spoof_detector, database)
    process(face_image)           # Process single face
    batch_process(face_images)    # Process multiple

Returns from process():
    {
        'timestamp': str,
        'success': bool,
        'error': str/None,
        'anti_spoof': dict,        # Anti-spoofing result
        'embedding_extracted': bool,
        'recognition': dict        # Recognition result
    }

Example Usage:
    pipeline = RecognitionPipeline(embedder, anti_spoof, db)
    result = pipeline.process(face_image)
    
    if result['success']:
        if result['anti_spoof']['is_real']:
            username = result['recognition']['username']
            similarity = result['recognition']['similarity']
            print(f"User: {username}, Sim: {similarity:.3f}")

"""

# ============================================================
# 4. face_detector.py
# ============================================================

"""
FaceDetector
- Detect wajah menggunakan Haar Cascade
- Returns (x, y, w, h, confidence) tuple untuk setiap face

Methods:
    __init__(model_path=None)      # Optional model path
    detect(frame)                  # Detect faces dalam frame
    detect_with_haar(frame)        # Direct Haar detection
    draw_detections(frame, detections, ...)  # Draw boxes

Returns from detect():
    list of tuples:
    [(x, y, w, h, confidence), ...]
    
    Where:
    - x, y: top-left corner
    - w, h: width, height
    - confidence: 1.0 untuk Haar (no score dari Haar)

Example Usage:
    detector = FaceDetector("models/haarcascade_frontalface_default.xml")
    faces = detector.detect(frame)
    
    for x, y, w, h, conf in faces:
        print(f"Face at ({x}, {y}), size: {w}x{h}")
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

"""

# ============================================================
# 5. main.py
# ============================================================

"""
SmartDoorLockApp
- Main application class
- Real-time face recognition dengan anti-spoofing

Methods:
    __init__()                    # Initialize all components
    warmup_camera()              # Stabilize camera
    process_frame(frame)         # Process single frame
    draw_hud(frame)             # Draw UI info
    run()                       # Main loop
    cleanup()                   # Release resources

Keyboard Controls:
    q - Quit
    e - Enroll new user

Display Output:
    - Spoofing status (REAL/FAKE)
    - Face bounding box
    - User name (if matched)
    - Similarity score
    - FPS counter
    - Timestamp
    
Example Usage:
    app = SmartDoorLockApp()
    success = app.run()

"""

# ============================================================
# 6. enrollment.py
# ============================================================

"""
PoseEnrollmentSystem
- Capture 5 pose wajah & extract embeddings
- Automatic anti-spoofing check
- Average embeddings untuk final representation

Methods:
    __init__()                           # Initialize components
    get_current_pose()                  # Get current pose info
    move_to_next_pose()                 # Advance ke next pose
    capture_pose(frame)                 # Capture single pose
    get_enrollments_summary()           # List captured poses
    finalize_enrollment()               # Average & normalize embeddings

Poses:
    1. 'front' - Depan (langsung ke kamera)
    2. 'left' - Kiri (miringkan ke kiri)
    3. 'right' - Kanan (miringkan ke kanan)
    4. 'up' - Atas (angkat kepala)
    5. 'down' - Bawah (turunkan kepala)

Flow:
    system = PoseEnrollmentSystem()
    
    for each frame:
        result = system.capture_pose(frame)
        if result['success']:
            print(f"Pose captured")
            system.move_to_next_pose()
    
    final_emb = system.finalize_enrollment()  # Average 5 embeddings
    database.add_enrollment(name, [final_emb])

"""

# ============================================================
# 7. config.py
# ============================================================

"""
Key Configuration Parameters:

Model Paths:
    MOBILEFACENET_PATH           # Path ke MobileFaceNet.tflite
    ANTI_SPOOFING_PATH           # Path ke FaceAntiSpoofing.tflite
    HAAR_CASCADE_PATH            # Path ke Haar Cascade

Anti-Spoofing:
    ANTI_SPOOF_THRESHOLD = 0.2         # Score threshold
    ANTI_SPOOF_LAPLACE_THRESHOLD = 50  # Blur variance threshold

Recognition:
    RECOGNITION_THRESHOLD = 0.7  # Cosine similarity threshold
    SIMILARITY_THRESHOLD = 0.7    # Same as above

Camera Settings:
    FRAME_WIDTH = 480/640        # Resolution (RPi vs Desktop)
    FRAME_HEIGHT = 360/480
    FPS = 20/30                  # Frames per second

Database:
    DB_NAME                      # Path ke face_database.db
    EMBEDDINGS_TABLE             # Table name

Environment:
    IS_RASPBERRY_PI              # Auto-detect RPi environment
    CAMERA_WARMUP_FRAMES         # Frames untuk warmup
    MEMORY_CLEANUP_INTERVAL      # Cleanup every N frames

"""

# ============================================================
# 8. Database Schema (LanceDB)
# ============================================================

"""
Face Database Schema

User Embeddings Table:
{
    'id': str,                      # Unique user ID
    'name': str,                    # Username
    'embedding': list[float],       # 128-dim embedding vector
    'created_at': str,              # ISO timestamp
    'last_seen': str,               # ISO timestamp
}

Methods:
    add_enrollment(name, embeddings)
    get_all_users()
    get_user_by_id(user_id)
    search_similar(embedding, top_k=5)
    
Example Usage:
    db = FaceDatabase()
    
    # Add new user
    db.add_enrollment("John Doe", [embedding_vector])
    
    # Get all users
    users = db.get_all_users()
    
    # Search similar embeddings
    matches = db.search_similar(query_embedding, top_k=3)

"""

# ============================================================
# WORKFLOW EXAMPLES
# ============================================================

"""
Complete Enrollment Workflow:

1. Initialize
from modules.face_detector import FaceDetector
from modules.embedder import FaceEmbedder
from modules.anti_spoofing import FaceAntiSpoofing
from modules.database import FaceDatabase

detector = FaceDetector()
embedder = FaceEmbedder("models/MobileFaceNet.tflite")
anti_spoof = FaceAntiSpoofing("models/FaceAntiSpoofing.tflite")
db = FaceDatabase()

2. Capture Frame & Detect Face
faces = detector.detect(frame)
x, y, w, h, conf = faces[0]
face_image = frame[y:y+h, x:x+w]

3. Anti-spoofing Check
spoof_result = anti_spoof.detect(face_image)
if not spoof_result['is_real']:
    print("Spoofing detected!")
    continue

4. Extract Embedding
embedding = embedder.extract_embedding(face_image)
embeddings.append(embedding)

5. Finalize (after 5 poses)
from modules.embedder import EmbeddingProcessor
final_emb = EmbeddingProcessor.average_embeddings(embeddings)

6. Save to Database
db.add_enrollment("John Doe", [final_emb])


Complete Recognition Workflow:

1. Initialize
from modules.recognition import RecognitionPipeline

pipeline = RecognitionPipeline(embedder, anti_spoof, db)

2. Process Face Image
result = pipeline.process(face_image)

3. Check Result
if result['success']:
    anti_spoof_status = result['anti_spoof']['label']  # REAL or FAKE
    
    if anti_spoof_status == 'REAL':
        username = result['recognition']['username']
        similarity = result['recognition']['similarity']
        print(f"{username}: {similarity:.3f}")

"""

# ============================================================
# PERFORMANCE TIPS
# ============================================================

"""
For Raspberry Pi 3:

1. Reduce Resolution
   FRAME_WIDTH = 320
   FRAME_HEIGHT = 240

2. Lower FPS
   FPS = 10

3. Skip Frames
   if frame_count % 2 == 0:
       # Process only even frames

4. Disable Display
   Skip cv2.imshow() dalam loop

5. Increase Thresholds
   ANTI_SPOOF_LAPLACE_THRESHOLD = 100  # Higher = allow more blur

6. Monitor Resources
   import psutil
   cpu_percent = psutil.cpu_percent(interval=1)

7. Cleanup Memory
   import gc
   gc.collect()  # Every 100 frames

For Desktop:

1. Increase Resolution
   FRAME_WIDTH = 1280
   FRAME_HEIGHT = 720

2. Higher FPS
   FPS = 30

3. Lower Thresholds
   ANTI_SPOOF_LAPLACE_THRESHOLD = 30  # More strict blur detection

4. Full Display
   Real-time visualization dengan FPS counter

"""

# ============================================================

Last Updated: 2026-04-05
