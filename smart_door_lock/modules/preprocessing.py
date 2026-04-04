"""
Image Preprocessing Module
- Normalisasi ukuran gambar
- Histogram equalization
- Kontras adjustment
- Brightness adjustment
"""

import cv2
import numpy as np
from config import TARGET_FACE_SIZE, FRAME_HEIGHT, FRAME_WIDTH, IS_RASPBERRY_PI

# Choose interpolation method based on platform
# RPi: use faster INTER_LINEAR; Desktop: use better quality INTER_CUBIC
INTERPOLATION_METHOD = cv2.INTER_LINEAR if IS_RASPBERRY_PI else cv2.INTER_CUBIC


class ImagePreprocessor:
    """Kelas untuk preprocessing image sebelum detection dan embedding"""
    
    def __init__(self):
        """Initialize preprocessor"""
        self.target_size = TARGET_FACE_SIZE
        
    def resize_frame(self, frame, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        """
        Resize frame ke ukuran standar - optimized untuk RPi
        
        Args:
            frame: Input frame dari camera
            width: Target width
            height: Target height
            
        Returns:
            Resized frame
        """
        return cv2.resize(frame, (width, height), interpolation=INTERPOLATION_METHOD)
    
    def crop_face(self, frame, bbox, padding=0.1):
        """
        Crop wajah dari frame berdasarkan bounding box
        
        Args:
            frame: Input frame
            bbox: Bounding box (x, y, w, h) atau (x1, y1, x2, y2)
            padding: Padding ratio (0-1)
            
        Returns:
            Cropped face image
        """
        if len(bbox) == 4:
            if bbox[2] > 1 or bbox[3] > 1:  # x1, y1, x2, y2 format
                x1, y1, x2, y2 = bbox
            else:  # x, y, w, h format
                x, y, w, h = bbox
                x1, y1, x2, y2 = x, y, x + w, y + h
        
        # Apply padding
        h, w = frame.shape[:2]
        x_pad = int((x2 - x1) * padding)
        y_pad = int((y2 - y1) * padding)
        
        x1 = max(0, x1 - x_pad)
        y1 = max(0, y1 - y_pad)
        x2 = min(w, x2 + x_pad)
        y2 = min(h, y2 + y_pad)
        
        cropped = frame[y1:y2, x1:x2]
        return cropped, (x1, y1, x2, y2)
    
    def normalize_face(self, face_img):
        """
        Normalize wajah ke ukuran standar dengan histogram equalization
        
        Args:
            face_img: Cropped face image
            
        Returns:
            Normalized face image
        """
        # Resize ke target size - optimized untuk RPi
        face_resized = cv2.resize(face_img, self.target_size, 
                                  interpolation=INTERPOLATION_METHOD)
        
        # Convert to grayscale untuk histogram equalization
        if len(face_resized.shape) == 3:
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_resized
        
        # Histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # Convert back to BGR if needed
        if len(face_resized.shape) == 3:
            equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            return equalized
        
        return equalized
    
    def enhance_contrast(self, image, alpha=1.2, beta=0):
        """
        Enhance kontras dan brightness
        
        Args:
            image: Input image
            alpha: Contrast factor (>1 meningkatkan contrast)
            beta: Brightness factor
            
        Returns:
            Enhanced image
        """
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return enhanced
    
    def prepare_for_detection(self, frame):
        """
        Prepare frame untuk face detection
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Resize
        frame = self.resize_frame(frame)
        
        # Enhance contrast untuk better detection
        frame = self.enhance_contrast(frame, alpha=1.1)
        
        return frame
    
    def prepare_for_embedding(self, face_img):
        """
        Prepare cropped face untuk embedding extraction
        
        Args:
            face_img: Cropped face image
            
        Returns:
            Preprocessed face image (112x112)
        """
        # Normalize size dan histogram
        normalized = self.normalize_face(face_img)
        
        # Normalize pixel values ke range [0, 1] atau [-1, 1]
        # ArcFace biasanya expect input dalam [0, 1]
        normalized = normalized.astype(np.float32) / 255.0
        
        return normalized
    
    @staticmethod
    def normalize_embedding(embedding):
        """
        L2 normalize embedding vector
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            Normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
