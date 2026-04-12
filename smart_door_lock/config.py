"""
Configuration untuk Smart Door Lock System
Python 3.9.6 Compatible - No directory creation at import time
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model paths - Haar Cascade untuk face detection
FACE_DETECTION_MODEL = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')
ANTI_SPOOFING_MODEL = os.path.join(MODELS_DIR, 'FaceAntiSpoofing.tflite')
EMBEDDING_MODEL = os.path.join(MODELS_DIR, 'MobileFaceNet.tflite')

# Database - SQLite (bukan pickle)
DATABASE_PATH = os.path.join(BASE_DIR, 'face_database.db')

# Anti-Spoofing Parameters
ANTI_SPOOFING = {
    'INPUT_IMAGE_SIZE': 256,
    'THRESHOLD': 0.2,  # < 0.2 = REAL
    'LAPLACE_THRESHOLD': 50,
    'LAPLACIAN_THRESHOLD': 1000,
}

# Face Detection Parameters (dari add_faces.py logic)
FACE_DETECTION = {
    'SCALE_FACTOR': 1.3,
    'MIN_NEIGHBORS': 5,
    'MIN_SIZE': (30, 30),
}

# Embedding Parameters
EMBEDDING = {
    'INPUT_SIZE': 112,
    'OUTPUT_SIZE': 128,
    'NORMALIZATION_MEAN': 0.5,
    'NORMALIZATION_STD': 0.5,
}

# Recognition Parameters
RECOGNITION = {
    'COSINE_THRESHOLD': 0.7,
}

# Enrollment Parameters - 20 REAL wajah (bukan 5 poses)
ENROLLMENT = {
    'TARGET_FACES': 20,
    'ANTI_SPOOFING_ENABLED': True,
}

# HUD Display Settings
HUD = {
    'FONT': 'cv2.FONT_HERSHEY_SIMPLEX',
    'FONT_SCALE': 0.7,
    'THICKNESS': 2,
    'COLOR_REAL': (0, 255, 0),      # Green
    'COLOR_FAKE': (0, 0, 255),      # Red
    'COLOR_MATCH': (0, 255, 0),     # Green
    'COLOR_NO_MATCH': (255, 0, 0),  # Red
    'COLOR_TEXT': (255, 255, 255),  # White
}

# Raspberry Pi Optimization
RASPBERRY_PI_MODE = False
MAX_FRAME_WIDTH = 640
MAX_FRAME_HEIGHT = 480
FPS_TARGET = 15
MEMORY_CLEANUP_INTERVAL = 100

