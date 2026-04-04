"""
Konfigurasi Global Smart Door Lock System
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Database
DB_NAME = os.path.join(DATA_DIR, 'face_database.db')
EMBEDDINGS_TABLE = 'face_embeddings'

# Model URLs
SCRFD_ONNX_URL = "https://huggingface.co/skytnt/anime-seg/resolve/main/face_detector.onnx"
ARCFACE_ONNX_URL = "https://huggingface.co/onnx-community/arcface-resnet100/resolve/main/model.onnx"
ARCFACE_WEIGHT_URL = "https://huggingface.co/skytnt/anime-seg/resolve/main/arcface.pth"

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Face Detection Settings
FACE_DETECT_CONFIDENCE = 0.5
FACE_MIN_SCALE = 50

# Face Preprocessing
TARGET_FACE_SIZE = (112, 112)  # ArcFace standard input
TARGET_FACE_SIZE_SCRFD = (480, 480)

# Embedding Settings
EMBEDDING_DIM = 512  # ArcFace embedding dimension
SIMILARITY_THRESHOLD = 0.6  # Threshold untuk accept wajah

# Enrollment Settings
ENROLLMENT_SAMPLES = 100
ENROLLMENT_SAMPLE_INTERVAL = 5  # Setiap frame ke-5

# Display Settings
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

# Create directories jika belum ada
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
