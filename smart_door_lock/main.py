"""
Smart Door Lock - Main Recognition Application
Real-time face recognition dengan anti-spoofing
Python 3.9.6 compatible - SQLite database
"""
import cv2
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import HUD
from core.face_detector import FaceDetector
from core.anti_spoofing import FaceAntiSpoofing
from core.embedder import FaceEmbedder
from core.recognition import RecognitionPipeline
from database import FaceDatabase
from enrollment import EnrollmentSystem


class SmartDoorLockApp:
    """Main Smart Door Lock Application"""
    
    def __init__(self):
        """Initialize Smart Door Lock App"""
        print("Initializing Smart Door Lock System...")
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.anti_spoofing = FaceAntiSpoofing()
        self.embedder = FaceEmbedder()
        
        # Initialize database
        self.db = FaceDatabase()
        self.database = self.db.get_all_users()
        
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
        
        enrollment_system = EnrollmentSystem(
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
                            enrollment_system.save_enrollment(result)
                            # Reload database
                            self.database = self.db.get_all_users()
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
