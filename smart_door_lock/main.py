"""
Smart Door Lock - Main Recognition Application
Real-time face recognition dengan anti-spoofing
"""
import cv2
import pickle
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATABASE_FILE, HUD
from core.face_detector import FaceDetector
from core.anti_spoofing import FaceAntiSpoofing
from core.embedder import FaceEmbedder
from core.recognition import RecognitionPipeline
from enrollment import PoseEnrollmentSystem


class SmartDoorLockApp:
    """Main Smart Door Lock Application"""
    
    def __init__(self, database_file=DATABASE_FILE):
        """
        Initialize Smart Door Lock App
        
        Args:
            database_file: Path ke face database pickle
        """
        print("Initializing Smart Door Lock System...")
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.anti_spoofing = FaceAntiSpoofing()
        self.embedder = FaceEmbedder()
        
        # Load database
        self.database = self.load_database(database_file)
        self.database_file = database_file
        
        # Create recognition pipeline
        self.pipeline = RecognitionPipeline(
            self.face_detector,
            self.anti_spoofing,
            self.embedder,
            self.database
        )
        
        # Statistics
        self.frame_count = 0
        self.face_detections = 0
        self.true_faces = 0
        self.matched_faces = 0
        
        print(f"✓ System initialized. Database: {len(self.database)} users")
    
    @staticmethod
    def load_database(database_file):
        """Load face database"""
        if os.path.exists(database_file):
            try:
                with open(database_file, 'rb') as f:
                    database = pickle.load(f)
                print(f"✓ Database loaded: {len(database)} registered users")
                return database
            except Exception as e:
                print(f"Warning: Error loading database: {e}")
                return {}
        else:
            print(f"Database not found: {database_file}")
            return {}
    
    def draw_hud(self, frame, results):
        """
        Draw HUD dengan recognition results
        
        Display:
        - Spoofing status (REAL / FAKE)
        - User name jika match
        - Similarity score
        - Face bounding boxes
        
        Args:
            frame: Input frame
            results: List of recognition results
            
        Returns:
            Frame dengan HUD
        """
        hud_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw header info
        cv2.putText(hud_frame, f"Frame: {self.frame_count}", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(hud_frame, f"Faces detected: {len(results)}", (15, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(hud_frame, f"Database users: {len(self.database)}", (15, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw footer (FPS info)
        if hasattr(self, 'fps'):
            cv2.putText(hud_frame, f"FPS: {self.fps:.1f}", (w-200, h-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw results untuk setiap face
        for idx, result in enumerate(results):
            x, y, w_face, h_face = result['face_coords']
            
            # Determine color based on spoofing status
            if result['is_real']:
                # REAL face
                rect_color = HUD['COLOR_REAL']  # Green
                status = "REAL"
            else:
                # FAKE face
                rect_color = HUD['COLOR_FAKE']  # Red
                status = "FAKE"
            
            # Draw bounding box
            cv2.rectangle(hud_frame, (x, y), (x+w_face, y+h_face), rect_color, 2)
            
            # Draw status text
            cv2.putText(hud_frame, status, (x, y-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, rect_color, 2)
            
            # Draw additional info if REAL
            if result['is_real'] and result['recognition_result']:
                recognition_result = result['recognition_result']
                
                if recognition_result['matched']:
                    # Matched face
                    user_id = recognition_result['user_id']
                    similarity = recognition_result['similarity']
                    
                    cv2.putText(hud_frame, f"User: {user_id}", (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD['COLOR_MATCH'], 2)
                    cv2.putText(hud_frame, f"Score: {similarity:.3f}", (x, y+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD['COLOR_MATCH'], 2)
                    cv2.putText(hud_frame, "MATCHED", (x, y+h_face+25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, HUD['COLOR_MATCH'], 3)
                    
                    self.matched_faces += 1
                else:
                    # Not matched
                    similarity = recognition_result['similarity']
                    
                    cv2.putText(hud_frame, f"No match", (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD['COLOR_NO_MATCH'], 2)
                    cv2.putText(hud_frame, f"Score: {similarity:.3f}", (x, y+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD['COLOR_NO_MATCH'], 2)
            
            if result['spoofing_result']:
                laplacian = result['spoofing_result']['laplacian_score']
                score = result['spoofing_result']['score']
                cv2.putText(hud_frame, f"L: {laplacian:.1f} S: {score:.3f}",
                           (x, y+h_face+50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            self.face_detections += 1
            if result['is_real']:
                self.true_faces += 1
        
        return hud_frame
    
    def run(self, video_capture=None, window_name="Smart Door Lock"):
        """
        Main recognition loop
        
        Keyboard controls:
        - 'q': Quit
        - 'e': Enter enrollment mode
        - 's': Save screenshot
        - 'r': Reset statistics
        
        Args:
            video_capture: OpenCV VideoCapture (default: camera 0)
            window_name: Window title
        """
        if video_capture is None:
            video_capture = cv2.VideoCapture(0)
            if not video_capture.isOpened():
                print("Error: Cannot open camera")
                return
        
        print("\n" + "="*60)
        print("Smart Door Lock - Recognition Engine")
        print("="*60)
        print("Controls:")
        print("  'q' : Quit")
        print("  'e' : Enrollment mode")
        print("  's' : Save screenshot")
        print("  'r' : Reset statistics")
        print("="*60 + "\n")
        
        enrollment_system = PoseEnrollmentSystem(
            self.face_detector,
            self.anti_spoofing,
            self.embedder
        )
        
        prev_frame_time = time.time()
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to read frame")
                break
            
            self.frame_count += 1
            
            # Process frame melalui pipeline
            results = self.pipeline.process_frame(frame)
            
            # Draw HUD
            display_frame = self.draw_hud(frame, results)
            
            # Calculate FPS
            current_time = time.time()
            self.fps = 1.0 / (current_time - prev_frame_time) if (current_time - prev_frame_time) > 0 else 0
            prev_frame_time = current_time
            
            # Show frame
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            
            elif key == ord('e'):
                print("\nEntering enrollment mode...")
                cv2.destroyAllWindows()
                
                user_id = input("Enter user ID to enroll: ").strip()
                if user_id:
                    result = enrollment_system.enroll_user(
                        video_capture, user_id, display_window=True
                    )
                    
                    if result['success']:
                        save = input("Save to database? (y/n): ").strip().lower()
                        if save == 'y':
                            enrollment_system.save_enrollment(result, self.database_file)
                            self.database = enrollment_system.load_database(self.database_file)
                            self.pipeline.database = self.database
                
                cv2.namedWindow(window_name)
            
            elif key == ord('s'):
                filename = f"screenshot_{self.frame_count}.png"
                cv2.imwrite(filename, display_frame)
                print(f"Screenshot saved: {filename}")
            
            elif key == ord('r'):
                self.frame_count = 0
                self.face_detections = 0
                self.true_faces = 0
                self.matched_faces = 0
                print("Statistics reset")
        
        # Cleanup
        video_capture.release()
        cv2.destroyAllWindows()
        
        # Print statistics
        print("\n" + "="*60)
        print("Statistics:")
        print(f"  Total frames processed: {self.frame_count}")
        print(f"  Total face detections: {self.face_detections}")
        print(f"  True faces (REAL): {self.true_faces}")
        print(f"  Matched faces: {self.matched_faces}")
        print("="*60 + "\n")


def main():
    """Main entry point"""
    try:
        app = SmartDoorLockApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
"""
Smart Door Lock - Main Application
Face Recognition + Anti-Spoofing System
TensorFlow Lite + Raspberry Pi 4 Optimized

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
import numpy as np
from datetime import datetime
import threading
import time

# Add current directory ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging to capture activities and errors in terminal
from logging_setup import setup_logging
logger = setup_logging()

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FPS,
    TEXT_COLOR, FAIL_COLOR, ERROR_COLOR, SHOW_FACE_BOX,
    CAMERA_WARMUP_FRAMES, MEMORY_CLEANUP_INTERVAL,
    IS_RASPBERRY_PI,
    MOBILEFACENET_PATH, ANTI_SPOOFING_PATH, HAAR_CASCADE_PATH,
    RECOGNITION_THRESHOLD, SIMILARITY_THRESHOLD,
    DATA_DIR
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
        logger.info("Initializing components...")
        
        try:
            self.face_detector = FaceDetector(HAAR_CASCADE_PATH)
            logger.info("Face detector initialized")
            
            self.embedder = FaceEmbedder(MOBILEFACENET_PATH)
            logger.info("Face embedder initialized")
            
            self.anti_spoofing = FaceAntiSpoofing(ANTI_SPOOFING_PATH)
            logger.info("Anti-spoofing detector initialized")
            
            self.database = FaceDatabase()
            logger.info("Database initialized")
            
            self.recognition_pipeline = RecognitionPipeline(
                self.embedder, 
                self.anti_spoofing,
                self.database
            )
            logger.info("Recognition pipeline initialized")
            
        except Exception:
            logger.exception("Failed to initialize components")
            raise
        
        # Initialize camera
        logger.info(f"Initializing camera {CAMERA_INDEX}...")
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        logger.info(f"Camera initialized: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS} FPS")
        
        # Storage untuk last recognition result
        self.last_result = None
        self.last_result_time = None
        
        logger.info("%s", "="*70)
        logger.info("SMART DOOR LOCK - FACE RECOGNITION + ANTI-SPOOFING")
        logger.info("%s", "="*70)
        logger.info(f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
        logger.info(f"Anti-spoofing: Enabled (Threshold: 0.2)")
        logger.info(f"Recognition threshold: {RECOGNITION_THRESHOLD}")
        logger.info(f"Mode: {'Raspberry Pi Optimized' if IS_RASPBERRY_PI else 'Desktop'}")
        logger.info("%s\n", "="*70)
    
    def warmup_camera(self):
        """Warmup camera untuk stabilisasi"""
        logger.info(f"Warming up camera ({CAMERA_WARMUP_FRAMES} frames)...")
        for i in range(CAMERA_WARMUP_FRAMES):
            ret, _ = self.cap.read()
            if not ret:
                raise RuntimeError("Camera warmup failed")
        logger.info("Camera ready")
    
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
                
                except Exception:
                    logger.exception("Processing face failed")
                
                # Draw face box
                if SHOW_FACE_BOX:
                    color = TEXT_COLOR
                    cv2.rectangle(frame_display, (x, y), (x+w, y+h), color, 2)
            
            # Draw recognition results
            self._draw_results(frame_display, recognized_faces)
            
            except Exception:
                logger.exception("Frame processing failed")
                cv2.putText(frame_display, f"Error: unexpected", (20, 50),
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
        except Exception:
            logger.exception("Camera warmup failed")
            return False
        
        logger.info("Starting face recognition")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    logger.error("Failed to read from camera")
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
                    logger.info("Quitting requested by user")
                    self.running = False
                elif key == ord('e'):
                    logger.info("Launching enrollment")
                    os.system(f'"{sys.executable}" enrollment.py')
                
                self.frame_count += 1
                
                # Memory cleanup
                if self.frame_count % MEMORY_CLEANUP_INTERVAL == 0:
                    gc.collect()
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        except Exception:
            logger.exception("Application error")
            return False
        
        finally:
            self.cleanup()
            return True
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        gc.collect()
        logger.info("Cleanup complete")


def main():
    """Main entry point"""
    try:
        app = SmartDoorLockApp()
        success = app.run()
        return success
    except Exception:
        logger.exception("Fatal error in application")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
