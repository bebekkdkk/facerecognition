"""
Face Detection Module
- Use Haar Cascade untuk deteksi wajah
- Simple dan reliable untuk Raspberry Pi
"""

import cv2
import numpy as np
from config import FACE_DETECT_CONFIDENCE, FACE_MIN_SCALE, HAAR_CASCADE_PATH


class FaceDetector:
    """Face detector menggunakan Haar Cascade - Optimized untuk RPi"""
    
    def __init__(self):
        """
        Initialize face detector dengan Haar Cascade lokal
        """
        try:
            self.haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
            
            # Check jika cascade loaded berhasil
            if self.haar_cascade.empty():
                print(f"[WARNING] Haar Cascade tidak bisa di-load dari: {HAAR_CASCADE_PATH}")
                print(f"[INFO] Fallback ke cascade default dari cv2.data")
                self.haar_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            else:
                print(f"[INFO] Haar Cascade loaded successfully: {HAAR_CASCADE_PATH}")
        except Exception as e:
            print(f"[ERROR] Failed to load Haar Cascade: {e}")
            # Fallback ke default
            self.haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
    def detect_with_haar(self, frame):
        """
        Detect wajah menggunakan Haar Cascade lokal
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of detections [(x, y, w, h, confidence), ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            faces = self.haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(FACE_MIN_SCALE, FACE_MIN_SCALE)
            )
            
            # Convert dari (x,y,w,h) ke [(x,y,w,h,conf), ...]
            # Haar Cascade tidak memberikan confidence score, default 1.0
            detections = [(x, y, w, h, 1.0) for x, y, w, h in faces]
            return detections
        except Exception as e:
            print(f"[ERROR] Haar Cascade detection failed: {e}")
            return []
    
    def detect(self, frame):
        """
        Detect wajah dalam frame - Gunakan Haar Cascade
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of detections dengan format (x, y, w, h, confidence)
        """
        return self.detect_with_haar(frame)
    
    def draw_detections(self, frame, detections, thickness=2, 
                       color=(0, 255, 0), show_confidence=True):
        """
        Draw bounding boxes pada frame
        
        Args:
            frame: Input frame
            detections: List of detections
            thickness: Line thickness
            color: BGR color tuple
            show_confidence: Show confidence score
            
        Returns:
            Frame dengan bounding boxes
        """
        frame_copy = frame.copy()
        
        for detection in detections:
            x, y, w, h, conf = detection
            
            # Draw rectangle
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color, thickness)
            
            # Draw confidence
            if show_confidence:
                label = f"{conf:.2f}"
                cv2.putText(frame_copy, label, (x, y-5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame_copy
