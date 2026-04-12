"""
Enrollment System - 20-Face Capture untuk Smart Door Lock
Python 3.9.6 compatible - No runtime_compat, SQLite database
"""
import cv2
import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ENROLLMENT, MODELS_DIR
from core.face_detector import FaceDetector
from core.anti_spoofing import FaceAntiSpoofing
from core.embedder import FaceEmbedder, EmbeddingProcessor
from database import FaceDatabase


class EnrollmentSystem:
    """
    20-Face automatic capture system untuk enrollment
    Menangkap 20 wajah real (anti-spoofing verified)
    Extract embedding dari MobileFaceNet
    Average semua embeddings menjadi single representation
    """
    
    def __init__(self, face_detector, anti_spoofing, embedder):
        """Initialize enrollment system"""
        self.face_detector = face_detector
        self.anti_spoofing = anti_spoofing
        self.embedder = embedder
        self.database = FaceDatabase()
        self.target_faces = ENROLLMENT.get('TARGET_FACES', 20)
    
    def enroll_user(self, video_capture, user_id, display_window=True):
        """
        Capture 20 real faces dan extract embeddings
        
        Flow:
        1. Capture frames continuously
        2. Detect wajah dalam setiap frame
        3. Check anti-spoofing (only accept REAL faces)
        4. Extract embedding dari MobileFaceNet
        5. Collect sampai 20 embeddings tercapai
        6. Average semua embeddings
        7. Save average ke SQLite database
        
        Args:
            video_capture: OpenCV VideoCapture object
            user_id: User identifier string
            display_window: Display live capture window
            
        Returns:
            Dict dengan 'success', 'embedding', 'message'
        """
        result = {
            'user_id': user_id,
            'embedding': None,
            'embeddings_count': 0,
            'success': False,
            'message': ''
        }
        
        all_embeddings = []
        frame_count = 0
        faces_captured = 0
        frame_skip = 5  # Sample setiap 5 frames
        
        print(f"\n{'='*60}")
        print(f"Starting enrollment untuk: {user_id}")
        print(f"Target: Capture {self.target_faces} real faces")
        print(f"Preview resolution: dari camera")
        print(f"Press 'q' untuk cancel enrollment")
        print(f"{'='*60}\n")
        
        while faces_captured < self.target_faces:
            ret, frame = video_capture.read()
            if not ret:
                result['message'] = 'Failed to read frame dari camera'
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Display progress
            cv2.putText(display_frame, f"Enrollment: {user_id}",
                       (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Progress: {faces_captured}/{self.target_faces}",
                       (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frames: {frame_count}",
                       (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Detect faces every frame_skip frames
            if frame_count % frame_skip == 0:
                faces = self.face_detector.detect(frame)
                
                if len(faces) > 0:
                    # Take largest face
                    face_coords = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = face_coords[:4]
                    
                    # Crop face image
                    face_img = frame[y:y+h, x:x+w]
                    
                    if face_img is not None and face_img.size > 0:
                        try:
                            # Anti-spoofing check FIRST
                            spoof_result = self.anti_spoofing.predict(face_img)
                            
                            if spoof_result.get('is_real', False):
                                # Extract embedding
                                embedding = self.embedder.extract_embedding(face_img)
                                
                                if embedding is not None:
                                    all_embeddings.append(embedding)
                                    faces_captured += 1
                                    
                                    # Draw rectangle (green untuk REAL)
                                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(display_frame, "REAL",
                                              (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
                                    print(f"✓ Face {faces_captured}/{self.target_faces} captured")
                            else:
                                # Draw rectangle (red untuk SPOOFING)
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                                cv2.putText(display_frame, "SPOOFING",
                                          (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        except Exception as e:
                            print(f"Error processing face: {e}")
                else:
                    cv2.putText(display_frame, "No face detected",
                               (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if display_window:
                cv2.imshow(f"Enrollment - {user_id}", display_frame)
            
            # Check keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                result['message'] = 'Enrollment cancelled by user'
                break
        
        if display_window:
            cv2.destroyAllWindows()
        
        # Average all embeddings jika berhasil capture
        if len(all_embeddings) >= self.target_faces:
            final_embedding = EmbeddingProcessor.average_embeddings(all_embeddings)
            result['embedding'] = final_embedding
            result['embeddings_count'] = len(all_embeddings)
            result['success'] = True
            result['message'] = f"Successfully enrolled {user_id} dengan {len(all_embeddings)} embeddings"
            
            print(f"\n{'='*60}")
            print(f"✓ Enrollment SUCCESSFUL!")
            print(f"  User: {user_id}")
            print(f"  Total embeddings: {len(all_embeddings)}")
            print(f"  Final embedding shape: {final_embedding.shape}")
            print(f"  Embedding method: Average dari MobileFaceNet")
            print(f"{'='*60}\n")
        else:
            result['success'] = False
            result['embeddings_count'] = len(all_embeddings)
            result['message'] = f"Failed: Only captured {len(all_embeddings)}/{self.target_faces} faces"
            print(f"\n✗ Enrollment FAILED: Only captured {len(all_embeddings)} faces\n")
        
        return result
    
    def save_enrollment(self, result):
        """
        Save enrollment ke SQLite database
        
        Args:
            result: Result dict dari enroll_user()
            
        Returns:
            bool (success/failure)
        """
        if not result['success'] or result['embedding'] is None:
            return False
        
        try:
            success = self.database.add_user(result['user_id'], result['embedding'])
            
            if success:
                count = self.database.get_user_count()
                print(f"✓ Database saved successfully")
                print(f"  User: {result['user_id']}")
                print(f"  Total users in database: {count}")
                return True
            else:
                print(f"✗ Failed to save user ke database")
                return False
                
        except Exception as e:
            print(f"✗ Error saving to database: {e}")
            return False


def main():
    """Main enrollment interface"""
    try:
        # Initialize components
        print("\nInitializing enrollment system...")
        face_detector = FaceDetector()
        anti_spoofing = FaceAntiSpoofing()
        embedder = FaceEmbedder()
        
        enrollment_system = EnrollmentSystem(
            face_detector,
            anti_spoofing,
            embedder
        )
        
        print("✓ All components initialized successfully\n")
        
        # Open camera
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            print("✗ Cannot open camera. Check device /dev/video0")
            return False
        
        # Get user ID from input
        user_id = input("Enter user ID untuk enroll: ").strip()
        if not user_id:
            print("✗ User ID cannot be empty")
            video.release()
            return False
        
        # Check if user already exists
        database = FaceDatabase()
        if database.user_exists(user_id):
            overwrite = input(f"User '{user_id}' bereits exists. Update? (y/n): ").strip().lower()
            if overwrite != 'y':
                video.release()
                return False
        
        # Start enrollment
        result = enrollment_system.enroll_user(video, user_id, display_window=True)
        
        # Save to database if successful
        if result['success']:
            save_prompt = input("\nSave enrollment to database? (y/n): ").strip().lower()
            if save_prompt == 'y':
                enrollment_system.save_enrollment(result)
            else:
                print("Enrollment not saved to database")
        
        video.release()
        cv2.destroyAllWindows()
        
        return result['success']
        
    except Exception as e:
        print(f"✗ Error in enrollment: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n✗ Enrollment interrupted by user")
        sys.exit(1)
