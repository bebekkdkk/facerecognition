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

# Database
DB_NAME = os.path.join(DATA_DIR, 'face_database.db')
EMBEDDINGS_TABLE = 'face_embeddings'

# Model URLs
SCRFD_ONNX_URL = "https://huggingface.co/skytnt/anime-seg/resolve/main/face_detector.onnx"
ARCFACE_ONNX_URL = "https://huggingface.co/onnx-community/arcface-resnet100/resolve/main/model.onnx"
ARCFACE_WEIGHT_URL = "https://huggingface.co/skytnt/anime-seg/resolve/main/arcface.pth"

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
TARGET_FACE_SIZE = (112, 112)  # ArcFace standard input
TARGET_FACE_SIZE_SCRFD = (480, 480)

# Embedding Settings
EMBEDDING_DIM = 512  # ArcFace embedding dimension
SIMILARITY_THRESHOLD = 0.6  # Threshold untuk accept wajah

# Enrollment Settings - Optimized untuk RPi
if IS_RASPBERRY_PI:
    ENROLLMENT_SAMPLES = 50  # RPi lebih cepat dengan sample lebih sedikit
    ENROLLMENT_SAMPLE_INTERVAL = 3  # Setiap frame ke-3 (faster)
else:
    ENROLLMENT_SAMPLES = 100
    ENROLLMENT_SAMPLE_INTERVAL = 5

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
