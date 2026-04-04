"""
Face Detection Module
- SCRFD via ONNX Runtime untuk deteksi wajah
- Fallback ke OpenCV Haar Cascade
"""

import cv2
import numpy as np
import onnxruntime as ort
from config import FACE_DETECT_CONFIDENCE, FACE_MIN_SCALE


class FaceDetector:
    """Face detector menggunakan SCRFD atau Haar Cascade"""
    
    def __init__(self, use_onnx=False, onnx_path=None):
        """
        Initialize face detector
        
        Args:
            use_onnx: Gunakan ONNX SCRFD model jika True
            onnx_path: Path ke SCRFD ONNX model
        """
        self.use_onnx = use_onnx and onnx_path is not None
        self.onnx_path = onnx_path
        self.onnx_session = None
        
        if self.use_onnx:
            try:
                self.onnx_session = ort.InferenceSession(
                    onnx_path,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                print(f"[INFO] SCRFD ONNX model loaded: {onnx_path}")
            except Exception as e:
                print(f"[WARNING] ONNX model loading failed: {e}")
                print("[INFO] Fallback ke Haar Cascade")
                self.use_onnx = False
        
        # Haar Cascade sebagai fallback
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
    def detect_with_haar(self, frame):
        """
        Detect wajah menggunakan Haar Cascade
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of detections [(x, y, w, h, confidence), ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(FACE_MIN_SCALE, FACE_MIN_SCALE)
        )
        
        # Convert dari (x,y,w,h) ke [(x,y,w,h,conf), ...]
        # Haar Cascade tidak memberikan confidence score
        detections = [(x, y, w, h, 1.0) for x, y, w, h in faces]
        return detections
    
    def detect_with_onnx(self, frame):
        """
        Detect wajah menggunakan SCRFD ONNX
        
        Args:
            frame: Input frame (BGR, 480x480)
            
        Returns:
            List of detections [(x1, y1, x2, y2, confidence), ...]
        """
        if self.onnx_session is None:
            return []
        
        try:
            # Preprocess
            img = cv2.resize(frame, (480, 480))
            img = (img - 127.5) / 128.0  # Normalize
            img = img.transpose(2, 0, 1)[np.newaxis, ...]  # HWC to NCHW
            img = img.astype(np.float32)
            
            # Inference
            input_name = self.onnx_session.get_inputs()[0].name
            outputs = self.onnx_session.run(None, {input_name: img})
            
            # Parse outputs
            # SCRFD outputs: [bboxes, landmarks, scores]
            bboxes = outputs[0][0]
            scores = outputs[2][0]
            
            # Filter oleh confidence
            detections = []
            h, w = frame.shape[:2]
            
            for bbox, score in zip(bboxes, scores):
                if score > FACE_DETECT_CONFIDENCE:
                    x1, y1, x2, y2 = bbox
                    # Scale ke frame size
                    x1 = int(x1 * w / 480)
                    y1 = int(y1 * h / 480)
                    x2 = int(x2 * w / 480)
                    y2 = int(y2 * h / 480)
                    
                    detections.append((x1, y1, x2, y2, float(score)))
            
            return detections
            
        except Exception as e:
            print(f"[ERROR] ONNX inference failed: {e}")
            return []
    
    def detect(self, frame):
        """
        Detect wajah dalam frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of detections dengan format (x, y, w, h, confidence)
        """
        if self.use_onnx:
            detections = self.detect_with_onnx(frame)
            # Convert dari (x1,y1,x2,y2) ke (x,y,w,h)
            detections = [
                (x1, y1, x2-x1, y2-y1, conf) 
                for x1, y1, x2, y2, conf in detections
            ]
            return detections
        else:
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
