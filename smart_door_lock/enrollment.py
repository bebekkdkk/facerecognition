"""
Enrollment Script - Face Enrollment Module
- Capture 5 pose wajah pengguna baru (front, left, right, up, down)
- Extract embedding untuk setiap pose
- Average embeddings menjadi single representation
- Simpan ke database
- Optimized untuk Raspberry Pi 3 + TensorFlow Lite
"""

import cv2
import sys
import os
import gc
import numpy as np
from datetime import datetime
import time

# Add parent directory ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, 
    TEXT_COLOR, FAIL_COLOR, DISPLAY_FPS,
    CAMERA_WARMUP_FRAMES, MEMORY_CLEANUP_INTERVAL,
    IS_RASPBERRY_PI,
    ENROLLMENT_POSES, ENROLLMENT_NUM_POSES,
    MOBILEFACENET_PATH, ANTI_SPOOFING_PATH,
    HAAR_CASCADE_PATH
)
from modules.face_detector import FaceDetector
from modules.embedder import FaceEmbedder, EmbeddingProcessor
from modules.anti_spoofing import FaceAntiSpoofing
from modules.database import FaceDatabase

class PoseEnrollmentSystem:
    """Multi-pose enrollment system untuk Raspberry Pi"""
    
    def __init__(self):
        """Initialize enrollment system"""
        self.face_detector = FaceDetector(HAAR_CASCADE_PATH)
        self.embedder = FaceEmbedder(MOBILEFACENET_PATH)
        self.anti_spoofing = FaceAntiSpoofing(ANTI_SPOOFING_PATH)
        self.database = FaceDatabase()
        
        # Pose tracking
        self.poses = list(ENROLLMENT_POSES.keys())
        self.current_pose_idx = 0
        self.pose_embeddings = {}  # {pose: embedding}
        self.all_embeddings = []
        
    def get_current_pose(self):
        """Get current pose info"""
        pose = self.poses[self.current_pose_idx]
        return pose, ENROLLMENT_POSES[pose]
    
    def move_to_next_pose(self):
        """Move ke next pose"""
        self.current_pose_idx += 1
        return self.current_pose_idx < len(self.poses)
    
    def capture_pose(self, frame):
        """
        Capture single pose
        
        Args:
            frame: Input frame
            
        Returns:
            dict: Capture result
        """
        result = {
            'success': False,
            'face_detected': False,
            'is_real': False,
            'embedding': None,
            'message': ''
        }
        
        # Detect face
        faces = self.face_detector.detect(frame)
        
        if len(faces) == 0:
            result['message'] = 'No face detected'
            return result
        
        result['face_detected'] = True
        
        # Get largest face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h, conf = face
        
        # Crop face
        face_image = frame[y:y+h, x:x+w]
        
        try:
            # Anti-spoofing check
            spoof_result = self.anti_spoofing.detect(face_image)
            
            if not spoof_result.get('is_real', False):
                result['message'] = 'SPOOFING DETECTED!'
                return result
            
            result['is_real'] = True
            
            # Extract embedding
            embedding = self.embedder.extract_embedding(face_image)
            result['embedding'] = embedding
            result['success'] = True
            result['message'] = 'CAPTURED!'
            
        except Exception as e:
            result['message'] = f'Error: {str(e)}'
        
        return result
    
    def get_enrollments_summary(self):
        """Get summary of captured embeddings"""
        summary = []
        for pose in self.poses:
            if pose in self.pose_embeddings:
                summary.append(pose)
        return summary
    
    def finalize_enrollment(self):
        """
        Finalize enrollment dengan average semua embeddings
        
        Returns:
            np.ndarray: Final consolidated embedding
        """
        if len(self.all_embeddings) == 0:
            raise ValueError("No embeddings captured")
        
        # Average semua embeddings
        final_embedding = EmbeddingProcessor.average_embeddings(self.all_embeddings)
        return final_embedding


def main():
    """Main enrollment process dengan 5-pose capture"""
    
    # Get username
    print("\n" + "="*60)
    print("SMART DOOR LOCK - MULTI-POSE ENROLLMENT (TFLite)")
    print("="*60)
    
    name = input("\n[INPUT] Enter your name: ").strip()
    if not name:
        print("[ERROR] Name cannot be empty!")
        return False
    
    # Initialize system
    print(f"\n[INFO] Initializing enrollment system for: {name}")
    
    try:
        enrollment_system = PoseEnrollmentSystem()
        print("[SUCCESS] All components initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize: {e}")
        return False
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"[ERROR] Camera {CAMERA_INDEX} is not accessible!")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Warmup camera
    print(f"\n[ACTION] Warming up camera ({CAMERA_WARMUP_FRAMES} frames)...")
    for i in range(CAMERA_WARMUP_FRAMES):
        ret, _ = cap.read()
        if not ret:
            print("[ERROR] Camera warmup failed!")
            cap.release()
            return False
    
    print("[ACTION] Camera ready. Position your face and follow the instructions.\n")
    
    try:
        frame_count = 0
        pose_start_time = time.time()
        pose_delay = 0
        
        while enrollment_system.current_pose_idx < len(enrollment_system.poses):
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read from camera!")
                break
            
            # Get current pose
            pose, pose_info = enrollment_system.get_current_pose()
            pose_delay = pose_info.get('delay', 0)
            
            # Display instructions
            frame_display = frame.copy()
            
            # Draw UI
            cv2.putText(frame_display, f"Pose {enrollment_system.current_pose_idx + 1}/{ENROLLMENT_NUM_POSES}: {pose_info['desc']}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2)
            
            # Capture delay countdown
            elapsed = time.time() - pose_start_time
            remaining_delay = max(0, pose_delay - elapsed)
            
            if remaining_delay > 0:
                cv2.putText(frame_display, f"Wait: {remaining_delay:.1f}s", 
                           (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Enrollment", frame_display)
                cv2.waitKey(100)
                frame_count += 1
                continue
            
            # Try to capture pose
            if frame_count % 3 == 0:  # Capture setiap 3 frames
                capture_result = enrollment_system.capture_pose(frame)
                
                # Update UI
                if capture_result['face_detected']:
                    status_color = TEXT_COLOR if capture_result['success'] else FAIL_COLOR
                    cv2.putText(frame_display, capture_result['message'], 
                               (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
                    
                    if capture_result['success']:
                        # Save embedding
                        enrollment_system.all_embeddings.append(capture_result['embedding'])
                        enrollment_system.pose_embeddings[pose] = capture_result['embedding']
                        
                        # Draw face box
                        faces = enrollment_system.face_detector.detect(frame)
                        if faces:
                            x, y, w, h, _ = faces[0]
                            cv2.rectangle(frame_display, (x, y), (x+w, y+h), TEXT_COLOR, 2)
                        
                        # Move to next pose
                        print(f"[CAPTURED] {pose.upper()}: Embedding saved ({len(enrollment_system.all_embeddings)}/5)")
                        
                        time.sleep(0.5)  # Brief pause before next pose
                        enrollment_system.move_to_next_pose()
                        pose_start_time = time.time()
                else:
                    cv2.putText(frame_display, "Position face in center", 
                               (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, FAIL_COLOR, 2)
            
            # Show frame
            cv2.imshow("Enrollment", frame_display)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[ABORT] Enrollment cancelled by user!")
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            frame_count += 1
            
            # Memory cleanup
            if frame_count % MEMORY_CLEANUP_INTERVAL == 0:
                gc.collect()
    
    except Exception as e:
        print(f"\n[ERROR] Error during enrollment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        gc.collect()
    
    # Check if all poses captured
    if len(enrollment_system.all_embeddings) < ENROLLMENT_NUM_POSES:
        print(f"\n[ERROR] Failed to capture all {ENROLLMENT_NUM_POSES} poses!")
        print(f"[INFO] Only captured {len(enrollment_system.all_embeddings)} poses.")
        return False
    
    # Finalize enrollment
    print("\n[PROCESSING] Finalizing enrollment...")
    
    try:
        final_embedding = enrollment_system.finalize_enrollment()
        
        # Save ke database
        success = enrollment_system.database.add_enrollment(name, [final_embedding])
        
        if success:
            print("\n" + "="*60)
            print(f"[SUCCESS] Enrollment completed for: {name}")
            print(f"[INFO] Captured poses: {', '.join(enrollment_system.get_enrollments_summary())}")
            print(f"[INFO] Final embedding: {len(final_embedding)} dimensions")
            print(f"[INFO] Enrollment method: Average of 5 poses + TFLite MobileFaceNet")
            print("="*60 + "\n")
            return True
        else:
            print(f"\n[ERROR] Failed to save to database!")
            return False
    
    except Exception as e:
        print(f"\n[ERROR] Failed to finalize enrollment: {e}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[ABORT] Program interrupted by user!")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
