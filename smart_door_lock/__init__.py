"""
Smart Door Lock Package
"""

__version__ = "1.0.0"
__author__ = "Smart Door Lock Team"

from .config import *
from .core import (
    FaceDetector,
    FaceAntiSpoofing,
    FaceEmbedder,
    EmbeddingProcessor,
    FaceRecognition,
    RecognitionPipeline
)
from .main import SmartDoorLockApp
from .enrollment import PoseEnrollmentSystem

__all__ = [
    'FaceDetector',
    'FaceAntiSpoofing',
    'FaceEmbedder',
    'EmbeddingProcessor',
    'FaceRecognition',
    'RecognitionPipeline',
    'SmartDoorLockApp',
    'PoseEnrollmentSystem',
]
