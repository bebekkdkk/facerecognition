"""
Face Detection menggunakan Haar Cascade
Logika dari add_faces.py
"""
import cv2
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FACE_DETECTION_MODEL, FACE_DETECTION


class FaceDetector:
    """Deteksi wajah menggunakan Haar Cascade - logika dari add_faces.py"""
    
    def __init__(self, model_path=FACE_DETECTION_MODEL):
        """
        Inisialisasi face detector
        
        Args:
            model_path: Path ke Haar Cascade XML
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Haar Cascade model not found: {model_path}")
        
        self.cascade = cv2.CascadeClassifier(model_path)
        if self.cascade.empty():
            raise ValueError(f"Failed to load Haar Cascade from {model_path}")
        
        self.model_path = model_path
    
    def detect(self, frame):
        """
        Deteksi wajah dalam frame (logika dari add_faces.py)
        
        Frame flow:
        1. Convert BGR to GRAY
        2. detectMultiScale dengan parameter dari config
        3. Return list of (x, y, w, h) tuples
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Array of (x, y, w, h) tuples
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION['SCALE_FACTOR'],  # 1.3
            minNeighbors=FACE_DETECTION['MIN_NEIGHBORS'],  # 5
            minSize=FACE_DETECTION['MIN_SIZE']  # (30, 30)
        )
        
        return faces
    
    def detect_with_confidence(self, frame):
        """
        Deteksi wajah dengan confidence score
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of (x, y, w, h, confidence) tuples
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces, weights = self.cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECTION['SCALE_FACTOR'],
            minNeighbors=FACE_DETECTION['MIN_NEIGHBORS'],
            outputRejectLevels=True,
            minSize=FACE_DETECTION['MIN_SIZE']
        )
        
        # Format: (x, y, w, h, confidence)
        result = []
        if len(weights) > 0:
            for i, (x, y, w, h) in enumerate(faces):
                # Normalize weight to confidence [0, 1]
                confidence = min(weights[0][i] / 10.0, 1.0)
                result.append((x, y, w, h, confidence))
        
        return result
    
    def crop_face(self, frame, face_coords):
        """
        Crop wajah dari frame (logika dari add_faces.py: frame[y:y+h, x:x+w, :])
        
        Args:
            frame: Input frame
            face_coords: (x, y, w, h) tuple
            
        Returns:
            Cropped face image atau None jika invalid
        """
        if face_coords is None or len(face_coords) < 4:
            return None
        
        x, y, w, h = face_coords[:4]
        
        # Validate coordinates
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            return None
        
        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            return None
        
        return frame[y:y+h, x:x+w, :]
    
    def resize_face(self, face_img, size=(50, 50)):
        """
        Resize wajah ke ukuran tertentu
        
        Args:
            face_img: Cropped face image
            size: Target size (width, height)
            
        Returns:
            Resized image
        """
        if face_img is None or face_img.size == 0:
            return None
        
        return cv2.resize(face_img, size)
    
    def draw_faces(self, frame, faces, color=(50, 50, 255), thickness=1):
        """
        Draw bounding boxes pada frame (logika dari add_faces.py)
        
        Args:
            frame: Input frame
            faces: Array of (x, y, w, h) tuples
            color: Warna rectangle (B, G, R)
            thickness: Ketebalan garis
            
        Returns:
            Frame dengan rectangles
        """
        frame_copy = frame.copy()
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color, thickness)
        
        return frame_copy
    
    def put_text_count(self, frame, count, position=(50, 50), 
                       font_scale=1, thickness=1, color=(50, 50, 255)):
        """
        Put text count pada frame (logika dari add_faces.py)
        
        Args:
            frame: Input frame
            count: Number to display
            position: (x, y) text position
            font_scale: Font scale
            thickness: Text thickness
            color: Text color (B, G, R)
            
        Returns:
            Frame dengan text
        """
        frame_copy = frame.copy()
        cv2.putText(frame_copy, str(count), position, 
                   cv2.FONT_HERSHEY_COMPLEX, font_scale, color, thickness)
        return frame_copy
