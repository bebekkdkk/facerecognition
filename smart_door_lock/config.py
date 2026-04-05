"""
Konfigurasi Global Smart Door Lock System
Optimized untuk Raspberry Pi
"""

import os
import platform

# Detect Raspberry Pi environment
IS_RASPBERRY_PI = os.path.exists('/proc/device-tree/model') or platform.machine() in ('armv7l', 'armv6l', 'aarch64')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Face Detection Models - Haar Cascade lokal
HAAR_CASCADE_PATH = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')

# TensorFlow Lite Models - MobileFaceNet & Anti-Spoofing
MOBILEFACENET_PATH = os.path.join(MODELS_DIR, 'MobileFaceNet.tflite')
ANTI_SPOOFING_PATH = os.path.join(MODELS_DIR, 'FaceAntiSpoofing.tflite')

# Database
DB_NAME = os.path.join(DATA_DIR, 'face_database.db')
EMBEDDINGS_TABLE = 'face_embeddings'

# Camera Settings - Optimized untuk RPi
CAMERA_INDEX = 0
if IS_RASPBERRY_PI:
    # RPi dengan keterbatasan resource, gunakan lower resolution
    FRAME_WIDTH = 480
    FRAME_HEIGHT = 360
    FPS = 20  # Lower FPS untuk reduce CPU
else:
    # Desktop/non-RPi systems
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30

# Face Detection Settings
FACE_DETECT_CONFIDENCE = 0.5
FACE_MIN_SCALE = 40 if IS_RASPBERRY_PI else 50

# Face Preprocessing
TARGET_FACE_SIZE = (112, 112)  # MobileFaceNet standard input

# Embedding Settings
EMBEDDING_DIM = 128  # MobileFaceNet embedding dimension
SIMILARITY_THRESHOLD = 0.7  # Threshold untuk match (cosine similarity)
RECOGNITION_THRESHOLD = 0.7  # Threshold untuk recognize

# Anti-Spoofing Settings
ANTI_SPOOF_INPUT_SIZE = 256
ANTI_SPOOF_THRESHOLD = 0.2  # Score < 0.2 = REAL
ANTI_SPOOF_LAPLACE_THRESHOLD = 50  # Blur detection threshold
ANTI_SPOOF_LAPLACIAN_THRESHOLD = 1000

# Enrollment Settings - Optimized untuk RPi
# Multi-pose enrollment: depan, kiri, kanan, atas, bawah
ENROLLMENT_POSES = {
    'front': {'desc': 'Depan (front)', 'delay': 0},
    'left': {'desc': 'Kiri (left)', 'delay': 2},
    'right': {'desc': 'Kanan (right)', 'delay': 2},
    'up': {'desc': 'Atas (up)', 'delay': 2},
    'down': {'desc': 'Bawah (down)', 'delay': 2}
}
ENROLLMENT_NUM_POSES = 5
ENROLLMENT_SAMPLES_PER_POSE = 1  # Save 1 embedding per pose
ENROLLMENT_TOTAL_EMBEDDINGS = 5  # Total 5 embeddings

# Display Settings - Optimized untuk RPi
if IS_RASPBERRY_PI:
    DISPLAY_FPS = False  # Disable FPS display pada RPi untuk save CPU
    DISPLAY_CONFIDENCE = True
else:
    DISPLAY_FPS = True
    DISPLAY_CONFIDENCE = True

SHOW_FACE_BOX = True
TEXT_COLOR = (0, 255, 0)  # BGR: Green
FAIL_COLOR = (0, 0, 255)  # BGR: Red
ERROR_COLOR = (0, 165, 255)  # BGR: Orange

# Logging
LOG_LEVEL = 'INFO'
LOG_FILE = os.path.join(DATA_DIR, 'smart_door.log')

# Access Control
MAX_ATTEMPTS = 3
ATTEMPT_TIMEOUT = 30  # detik

# Performance Settings untuk RPi
CAMERA_WARMUP_FRAMES = 10  # Frames untuk camera warmup
MEMORY_CLEANUP_INTERVAL = 100  # Cleanup every N frames

# Create directories jika belum ada
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Print configuration info
if IS_RASPBERRY_PI:
    print(f"[CONFIG] Running on Raspberry Pi - optimized settings enabled")
    print(f"[CONFIG] Frame size: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"[CONFIG] Enrollment samples: {ENROLLMENT_SAMPLES}")
