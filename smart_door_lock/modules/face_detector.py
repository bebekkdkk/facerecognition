"""
Face Detector using OpenCV Haar Cascade.

Provides a simple `FaceDetector` wrapper that returns detected faces
in format: (x, y, w, h, confidence).

This implementation is lightweight and suitable for Raspberry Pi 4.
"""

import os
import cv2
import numpy as np
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HAAR_CASCADE_PATH, FACE_MIN_SCALE


class FaceDetector:
    """Haar Cascade face detector wrapper."""

    def __init__(self, cascade_path=None, min_size=None, scale_factor=1.1, min_neighbors=5):
        """Initialize detector.

        Args:
            cascade_path (str): Path to Haar cascade XML file.
            min_size (tuple|int): Minimum face size (w,h) or int for square.
            scale_factor (float): Scale factor for detectMultiScale.
            min_neighbors (int): minNeighbors for detectMultiScale.
        """
        if cascade_path is None:
            cascade_path = HAAR_CASCADE_PATH

        # Fallback to OpenCV's bundled cascades if file missing
        if not os.path.exists(cascade_path):
            default = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            cascade_path = default

        self.cascade_path = cascade_path
        self.detector = cv2.CascadeClassifier(self.cascade_path)

        if min_size is None:
            min_size = (FACE_MIN_SCALE, FACE_MIN_SCALE)
        elif isinstance(min_size, int):
            min_size = (min_size, min_size)

        self.min_size = tuple(min_size)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, image):
        """Detect faces in an image.

        Args:
            image (np.ndarray): BGR image from OpenCV.

        Returns:
            list of tuples: [(x, y, w, h, confidence), ...]
        """
        if image is None:
            return []

        # Convert to grayscale for Haar detector
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size
        )

        results = []
        img_area = max(1, image.shape[0] * image.shape[1])
        for (x, y, w, h) in faces:
            # Heuristic confidence: proportional to face area relative to image
            area = w * h
            conf = min(1.0, (area / img_area) * 5.0)
            results.append((int(x), int(y), int(w), int(h), float(conf)))

        return results
