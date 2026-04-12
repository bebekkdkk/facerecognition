"""
Enrollment System - 5-Pose Automatic Face Capture dan Embedding Extraction
"""
import cv2
import pickle
import numpy as np
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATABASE_FILE, ENROLLMENT
from core.face_detector import FaceDetector
from core.anti_spoofing import FaceAntiSpoofing
from core.embedder import FaceEmbedder, EmbeddingProcessor


class PoseEnrollmentSystem:
    """
    5-Pose capture system untuk enrollment
    
    Poses:
    1. FRONT - straight face
    2. LEFT - turn head left
    3. RIGHT - turn head right
    4. UP - look up
    5. DOWN - look down
    """
    
    POSES = ['FRONT', 'LEFT', 'RIGHT', 'UP', 'DOWN']
    
    def __init__(self, face_detector, anti_spoofing, embedder, 
                 frames_per_pose=5, max_frames_per_pose=100):
        """
        Inisialisasi enrollment system
        
        Args:
            face_detector: FaceDetector instance
            anti_spoofing: FaceAntiSpoofing instance
            embedder: FaceEmbedder instance
            frames_per_pose: Target frames per pose to capture
            max_frames_per_pose: Maximum frames per pose (100 untuk user)
        """
        self.face_detector = face_detector
        self.anti_spoofing = anti_spoofing
        self.embedder = embedder
        
        self.frames_per_pose = frames_per_pose
        self.max_frames_per_pose = max_frames_per_pose
    
    def enroll_user(self, video_capture, user_id, display_window=True):
        """
        Capture 5 poses dan extract embeddings
        
        Flow:
        1. untuk setiap pose (5):
           a. Capture frames sampai max_frames_per_pose
           b. Every 10 frames (frame_skip), ambil wajah yang REAL (anti-spoofing check)
           c. Extract embedding dari wajah yang valid
           d. Continue sampai cukup embeddings atau max frames tercapai
        2. Average semua embeddings (EmbeddingProcessor.average_embeddings)
        3. Return final 128-dim embedding (L2-normalized)
        
        Args:
            video_capture: OpenCV VideoCapture object
            user_id: User identifier string
            display_window: Display live capture window (bool)
            
        Returns:
            Dict {
                'user_id': str,
                'embedding': 128-dim array,
                'embeddings_per_pose': dict {pose: list of embeddings},
                'success': bool,
                'message': str
            }
        """
        result = {
            'user_id': user_id,
            'embedding': None,
            'embeddings_per_pose': {},
            'success': False,
            'message': ''
        }
        
        all_embeddings = []
        frame_skip = 10  # Sample setiap 10 frames (seperti logika add_faces.py)
        
        for pose_idx, pose in enumerate(self.POSES):
            print(f"\n{'='*50}")
            print(f"Pose {pose_idx + 1}/5: {pose}")
            print(f"{'='*50}")
            print(f"Instructions:")
            print(f"  FRONT: Look straight at camera")
            print(f"  LEFT: Turn head to the left")
            print(f"  RIGHT: Turn head to the right")
            print(f"  UP: Look up at camera")
            print(f"  DOWN: Look down at camera")
            print(f"\nCapturing up to {self.max_frames_per_pose} frames...")
            print(f"Press 'q' to skip this pose or when done")
            
            pose_embeddings = []
            frame_count = 0
            captured_count = 0
            real_count = 0
            
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    print("Error: Failed to read frame")
                    break
                
                frame_count += 1
                
                # Display info
                info_frame = frame.copy()
                cv2.putText(info_frame, f"Pose: {pose} ({pose_idx + 1}/5)",
                           (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(info_frame, f"Frames: {frame_count}/{self.max_frames_per_pose}",
                           (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(info_frame, f"Captured: {captured_count}",
                           (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(info_frame, f"Real faces: {real_count}",
                           (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Detect faces
                faces = self.face_detector.detect(frame)
                
                if len(faces) > 0 and frame_count % frame_skip == 0:
                    # Take first face
                    face_coords = faces[0]
                    face_img = self.face_detector.crop_face(frame, face_coords)
                    
                    if face_img is not None and face_img.size > 0:
                        # Anti-spoofing check FIRST
                        spoofing_result = self.anti_spoofing.predict(face_img)
                        
                        if spoofing_result['is_real']:
                            # Extract embedding
                            embedding = self.embedder.extract_embedding(face_img)
                            
                            if embedding is not None:
                                pose_embeddings.append(embedding)
                                all_embeddings.append(embedding)
                                captured_count += 1
                                real_count += 1
                                
                                # Draw rectangle (green untuk REAL)
                                x, y, w, h = face_coords
                                cv2.rectangle(info_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(info_frame, "REAL", (x, y-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            # Draw rectangle (red untuk FAKE)
                            x, y, w, h = face_coords
                            cv2.rectangle(info_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            cv2.putText(info_frame, "FAKE", (x, y-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                elif len(faces) > 0:
                    # Draw detected faces (sampling)
                    x, y, w, h = faces[0]
                    cv2.rectangle(info_frame, (x, y), (x+w, y+h), (255, 255, 0), 1)
                
                if display_window:
                    cv2.imshow(f"Enrollment - {pose}", info_frame)
                
                # Check stop conditions
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or frame_count >= self.max_frames_per_pose:
                    break
                
                if captured_count >= self.frames_per_pose:
                    print(f"  ✓ Captured {captured_count} real faces, moving to next pose...")
                    break
            
            result['embeddings_per_pose'][pose] = pose_embeddings
            
            print(f"  Total valid embeddings captured for {pose}: {len(pose_embeddings)}")
            
            if display_window:
                cv2.destroyAllWindows()
        
        # Average all embeddings
        if len(all_embeddings) > 0:
            final_embedding = EmbeddingProcessor.average_embeddings(all_embeddings)
            result['embedding'] = final_embedding
            result['success'] = True
            result['message'] = f"Successfully enrolled {user_id} with {len(all_embeddings)} total embeddings"
            print(f"\n{'='*50}")
            print(f"✓ Enrollment successful!")
            print(f"  User ID: {user_id}")
            print(f"  Total embeddings captured: {len(all_embeddings)}")
            print(f"  Final embedding shape: {final_embedding.shape}")
            print(f"{'='*50}\n")
        else:
            result['success'] = False
            result['message'] = f"Failed to capture any valid embeddings for {user_id}"
            print(f"\n✗ Enrollment failed: No valid embeddings captured")
        
        return result
    
    def save_enrollment(self, result, database_file=DATABASE_FILE):
        """
        Save enrollment ke database pickle file
        
        Args:
            result: Result dict dari enroll_user()
            database_file: Path ke database file
            
        Returns:
            bool (success/failure)
        """
        if not result['success'] or result['embedding'] is None:
            return False
        
        try:
            # Load existing database atau buat baru
            if os.path.exists(database_file):
                with open(database_file, 'rb') as f:
                    database = pickle.load(f)
            else:
                database = {}
            
            # Add/update user
            database[result['user_id']] = result['embedding']
            
            # Save
            os.makedirs(os.path.dirname(database_file), exist_ok=True)
            with open(database_file, 'wb') as f:
                pickle.dump(database, f)
            
            print(f"✓ Database saved: {database_file}")
            print(f"  Total users in database: {len(database)}")
            return True
            
        except Exception as e:
            print(f"✗ Error saving database: {e}")
            return False
    
    @staticmethod
    def load_database(database_file=DATABASE_FILE):
        """
        Load database dari pickle file
        
        Args:
            database_file: Path ke database file
            
        Returns:
            Dict {user_id: embedding} atau {} jika file tidak ada
        """
        if os.path.exists(database_file):
            try:
                with open(database_file, 'rb') as f:
                    database = pickle.load(f)
                print(f"✓ Database loaded: {len(database)} users")
                return database
            except Exception as e:
                print(f"✗ Error loading database: {e}")
                return {}
        else:
            print(f"Database file not found: {database_file}")
            return {}


def main_enrollment():
    """Main enrollment loop"""
    try:
        # Initialize components
        print("Initializing enrollment system...")
        face_detector = FaceDetector()
        anti_spoofing = FaceAntiSpoofing()
        embedder = FaceEmbedder()
        
        # Open camera
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            print("Error: Cannot open camera")
            return
        
        # Create enrollment system
        enrollment_system = PoseEnrollmentSystem(
            face_detector,
            anti_spoofing,
            embedder,
            frames_per_pose=ENROLLMENT['FRAMES_PER_POSE'],
            max_frames_per_pose=ENROLLMENT['MAX_FRAMES_PER_POSE']
        )
        
        # Get user ID
        user_id = input("\nEnter user ID to enroll: ").strip()
        if not user_id:
            print("Error: User ID cannot be empty")
            return
        
        # Start enrollment
        result = enrollment_system.enroll_user(video, user_id, display_window=True)
        
        # Save to database
        if result['success']:
            save = input("\nSave to database? (y/n): ").strip().lower()
            if save == 'y':
                enrollment_system.save_enrollment(result)
        
        video.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error in enrollment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main_enrollment()
"""
Enrollment Script - Face Enrollment Module
- Capture 5 pose wajah pengguna baru (front, left, right, up, down)
- Extract embedding untuk setiap pose
- Average embeddings menjadi single representation
- Simpan ke database
- Optimized untuk Raspberry Pi 4 (Model B) + TensorFlow Lite
"""

import runtime_compat  # noqa: F401 - apply runtime env guards early

import cv2
import sys
import os
import gc
import numpy as np
from datetime import datetime
import time

# Add parent directory ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Setup logging to show activities and errors in terminal
from logging_setup import setup_logging
logger = setup_logging()

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

if IS_RASPBERRY_PI:
    cv2.setNumThreads(1)

class PoseEnrollmentSystem:
    """Multi-pose enrollment system untuk Raspberry Pi 4 and similar ARM devices"""
    
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
    
    except Exception:
        logger.exception("Error during enrollment")
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
        logger.info("Enrollment aborted by user")
        sys.exit(1)
    except Exception:
        logger.exception("Unexpected error in enrollment")
        sys.exit(1)
