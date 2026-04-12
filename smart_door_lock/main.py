"""
Smart Door Lock - Main Application
Face Recognition + Anti-Spoofing System
TensorFlow Lite + Raspberry Pi 3 Optimized

Main entry point untuk:
- Real-time face recognition dengan anti-spoofing check
- Display hasil recognition dengan similarity score
- Access control dan logging
"""

import runtime_compat  # noqa: F401 - apply runtime env guards early

import cv2
import sys
import os
import gc
import subprocess
from datetime import datetime

# Add current directory ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS,
    TEXT_COLOR, FAIL_COLOR, ERROR_COLOR, SHOW_FACE_BOX,
    CAMERA_WARMUP_FRAMES, MEMORY_CLEANUP_INTERVAL,
    IS_RASPBERRY_PI,
    MOBILEFACENET_PATH, ANTI_SPOOFING_PATH, HAAR_CASCADE_PATH,
    RECOGNITION_THRESHOLD
)
from modules.face_detector import FaceDetector
from modules.embedder import FaceEmbedder
from modules.anti_spoofing import FaceAntiSpoofing
from modules.database import FaceDatabase
from modules.recognition import RecognitionPipeline

if IS_RASPBERRY_PI:
    cv2.setNumThreads(1)


class SmartDoorLockApp:
    """Main application untuk face recognition + anti-spoofing"""
    
    def __init__(self):
        """Initialize application"""
        self.running = True
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_timer = datetime.now()
        
        # Initialize components
        print("[INFO] Initializing components...")
        
        try:
            self.face_detector = FaceDetector(HAAR_CASCADE_PATH)
            print("[SUCCESS] Face detector initialized")
            
            self.embedder = FaceEmbedder(MOBILEFACENET_PATH)
            print("[SUCCESS] Face embedder initialized")
            
            self.anti_spoofing = FaceAntiSpoofing(ANTI_SPOOFING_PATH)
            print("[SUCCESS] Anti-spoofing detector initialized")
            
            self.database = FaceDatabase()
            print("[SUCCESS] Database initialized")
            
            self.recognition_pipeline = RecognitionPipeline(
                self.embedder, 
                self.anti_spoofing,
                self.database
            )
            print("[SUCCESS] Recognition pipeline initialized")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize components: {e}")
            raise
        
        # Initialize camera
        print(f"\n[INFO] Initializing camera {CAMERA_INDEX}...")
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"[SUCCESS] Camera initialized: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS} FPS")
        
        # Storage untuk last recognition result
        self.last_result = None
        self.last_result_time = None
        
        print("\n" + "="*70)
        print("SMART DOOR LOCK - FACE RECOGNITION + ANTI-SPOOFING")
        print("="*70)
        print(f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
        print(f"Anti-spoofing: Enabled (Threshold: 0.2)")
        print(f"Recognition threshold: {RECOGNITION_THRESHOLD}")
        print(f"Mode: {'Raspberry Pi Optimized' if IS_RASPBERRY_PI else 'Desktop'}")
        print("="*70 + "\n")
    
    def warmup_camera(self):
        """Warmup camera untuk stabilisasi"""
        print(f"[ACTION] Warming up camera ({CAMERA_WARMUP_FRAMES} frames)...")
        for i in range(CAMERA_WARMUP_FRAMES):
            ret, _ = self.cap.read()
            if not ret:
                raise RuntimeError("Camera warmup failed")
        print("[SUCCESS] Camera ready\n")
    
    def process_frame(self, frame):
        """
        Process single frame
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            tuple: (frame_display, recognized_faces)
        """
        frame_display = frame.copy()
        recognized_faces = []
        
        try:
            # Detect faces
            faces = self.face_detector.detect(frame)
            
            if len(faces) == 0:
                # No face detected
                cv2.putText(frame_display, "No face detected", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, ERROR_COLOR, 2)
                return frame_display, recognized_faces
            
            # Process setiap detected face
            for face in faces:
                x, y, w, h, conf = face
                
                # Crop face
                face_image = frame[max(0, y):min(frame.shape[0], y+h),
                                   max(0, x):min(frame.shape[1], x+w)]
                
                if face_image.size == 0:
                    continue
                
                try:
                    # Run recognition pipeline
                    result = self.recognition_pipeline.process(face_image)
                    
                    if result.get('success'):
                        recognized_faces.append({
                            'location': (x, y, w, h),
                            'result': result
                        })
                        
                        # Store last result
                        self.last_result = result
                        self.last_result_time = datetime.now()
                
                except Exception as e:
                    print(f"[ERROR] Processing face failed: {e}")
                
                # Draw face box
                if SHOW_FACE_BOX:
                    color = TEXT_COLOR
                    cv2.rectangle(frame_display, (x, y), (x+w, y+h), color, 2)
            
            # Draw recognition results
            self._draw_results(frame_display, recognized_faces)
            
        except Exception as e:
            print(f"[ERROR] Frame processing failed: {e}")
            cv2.putText(frame_display, f"Error: {str(e)}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, ERROR_COLOR, 2)
        
        return frame_display, recognized_faces
    
    def _draw_results(self, frame, recognized_faces):
        """Draw recognition results on frame"""
        for face_data in recognized_faces:
            x, y, w, h = face_data['location']
            result = face_data['result']
            
            # Get recognition info
            recognition = result.get('recognition', {})
            anti_spoof = result.get('anti_spoof', {})
            
            # Draw anti-spoofing status
            spoofing_status = anti_spoof.get('label', 'UNKNOWN')
            spoofing_color = TEXT_COLOR if spoofing_status == 'REAL' else FAIL_COLOR
            
            cv2.putText(frame, f"Spoof: {spoofing_status}", (x, y - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, spoofing_color, 1)
            
            # Draw identity
            username = recognition.get('username', 'UNKNOWN')
            similarity = recognition.get('similarity', 0.0)
            matched = recognition.get('matched', False)
            
            identity_color = TEXT_COLOR if matched else FAIL_COLOR
            
            cv2.putText(frame, f"{username}", (x, y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, identity_color, 2)
            
            cv2.putText(frame, f"Sim: {similarity:.3f}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, identity_color, 1)
    
    def draw_hud(self, frame):
        """Draw heads-up display dengan FPS dan info"""
        h, w = frame.shape[:2]
        
        # Draw timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
        
        # Draw FPS (tapi hanya jika bukan RPi untuk save CPU)
        if not IS_RASPBERRY_PI:
            self.fps_counter += 1
            elapsed = (datetime.now() - self.fps_timer).total_seconds()
            if elapsed >= 1.0:
                fps = self.fps_counter / elapsed
                cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
                self.fps_counter = 0
                self.fps_timer = datetime.now()
        
        # Draw mode indicator
        mode = "RPi" if IS_RASPBERRY_PI else "Desktop"
        cv2.putText(frame, f"Mode: {mode}", (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    def run(self):
        """Main loop aplikasi"""
        # Warmup camera
        try:
            self.warmup_camera()
        except Exception as e:
            print(f"[ERROR] Camera warmup failed: {e}")
            return False
        
        print("[ACTION] Starting face recognition...\n")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("[ERROR] Failed to read from camera!")
                    break
                
                # Process frame
                frame_display, recognized_faces = self.process_frame(frame)
                
                # Draw HUD
                self.draw_hud(frame_display)
                
                # Display
                cv2.imshow("Smart Door Lock", frame_display)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] Quitting...")
                    self.running = False
                elif key == ord('e'):
                    print("\n[INFO] Press 'Enter' to start enrollment")
                    enrollment_script = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "enrollment.py"
                    )
                    subprocess.call([sys.executable, enrollment_script])
                
                self.frame_count += 1
                
                # Memory cleanup
                if self.frame_count % MEMORY_CLEANUP_INTERVAL == 0:
                    gc.collect()
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        
        except Exception as e:
            print(f"\n[ERROR] Application error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            self.cleanup()
            return True
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n[ACTION] Cleaning up...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        gc.collect()
        print("[SUCCESS] Cleanup complete")


def main():
    """Main entry point"""
    try:
        app = SmartDoorLockApp()
        success = app.run()
        return success
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
