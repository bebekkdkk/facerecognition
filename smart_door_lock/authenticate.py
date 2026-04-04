"""
Authentication/Access Control Script
- Capture wajah
- Lakukan identification menggunakan embedding + vector search
- Decision logic dan access control
- Log aktivitas
"""

import cv2
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    TEXT_COLOR, FAIL_COLOR, ERROR_COLOR,
    DISPLAY_FPS, SIMILARITY_THRESHOLD,
    MAX_ATTEMPTS, ATTEMPT_TIMEOUT,
    DATA_DIR
)
from modules import (
    ImagePreprocessor, FaceDetector, FaceEmbedder,
    FaceDatabase, FaceTracker
)


class SmartDoorLock:
    """Smart Door Lock Authentication System"""
    
    def __init__(self):
        """Initialize door lock system"""
        self.preprocessor = ImagePreprocessor()
        self.detector = FaceDetector(use_onnx=False)
        self.embedder = FaceEmbedder()
        self.database = FaceDatabase()
        self.tracker = FaceTracker()
        
        self.log_file = os.path.join(DATA_DIR, 'access_log.json')
        self.cap = None
        
        print("[INFO] Smart Door Lock System initialized")
    
    def log_access(self, status, name, similarity=None, details=None):
        """
        Log akses attempt
        
        Args:
            status: 'GRANTED' atau 'DENIED'
            name: Nama pengguna (atau 'UNKNOWN')
            similarity: Similarity score
            details: Additional details
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "name": name,
            "similarity": float(similarity) if similarity else None,
            "details": details
        }
        
        # Append ke log file
        try:
            logs = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            
            logs.append(log_entry)
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to log access: {e}")
    
    def authenticate_face(self, embedding):
        """
        Authenticate wajah menggunakan embedding
        
        Args:
            embedding: Face embedding vector
            
        Returns:
            (status, name, similarity) dimana status = 'GRANTED' atau 'DENIED'
        """
        # Search dalam database
        matches = self.database.search_similar(
            embedding,
            top_k=1,
            threshold=SIMILARITY_THRESHOLD
        )
        
        if matches:
            best_match = matches[0]
            name = best_match['name']
            similarity = best_match['similarity']
            
            print(f"\n[MATCH] Found: {name}")
            print(f"[MATCH] Similarity: {similarity:.4f}")
            
            return 'GRANTED', name, similarity
        else:
            print(f"\n[NO_MATCH] Face not recognized")
            return 'DENIED', 'UNKNOWN', 0.0
    
    def run(self):
        """Main authentication loop"""
        
        print("\n" + "="*60)
        print("SMART DOOR LOCK - ACCESS CONTROL")
        print("="*60)
        
        # Check database
        stats = self.database.get_stats()
        if stats['total_users'] == 0:
            print("[WARNING] No enrolled users in database!")
            print("[INFO] Please run enrollment.py first")
            return
        
        print(f"[INFO] Enrolled users: {', '.join(stats['users'])}")
        print("\n[ACTION] Face scanning in progress... Press 'q' to exit\n")
        
        # Open camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        # Tracking variables
        confirmed_identity = None
        confirmed_similarity = 0.0
        confidence_threshold = 3  # Frames untuk confirm identity
        frame_count = 0
        fps_counter = 0
        fps_timer = datetime.now()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to read from camera!")
                break
            
            # Preprocess
            frame = self.preprocessor.resize_frame(frame)
            
            # Detect faces
            detections = self.detector.detect(frame)
            
            # Update tracker
            embeddings = []
            for detection in detections:
                x, y, w, h, conf = detection
                if conf < 0.5:
                    continue
                
                # Crop dan preprocess
                face_region = frame[y:y+h, x:x+w]
                face_prepared = self.preprocessor.prepare_for_embedding(face_region)
                
                # Extract embedding
                embedding = self.embedder.extract(face_prepared)
                embeddings.append(embedding)
            
            tracked = self.tracker.update(detections, embeddings)
            
            # Draw frame
            frame_display = self.detector.draw_detections(
                frame, detections,
                color=TEXT_COLOR
            )
            
            # Process tracked faces
            for track_id, track_data in tracked.items():
                embedding = track_data['embedding']
                frames = track_data['frames']
                
                # Authenticate jika sudah stable
                if frames >= confidence_threshold and embedding is not None:
                    status, name, similarity = self.authenticate_face(embedding)
                    
                    if status == 'GRANTED':
                        confirmed_identity = name
                        confirmed_similarity = similarity
                        
                        # Log success
                        self.log_access(status, name, similarity)
                        
                        # Show success message
                        color = TEXT_COLOR
                        msg = f"ACCESS GRANTED: {name}"
                    else:
                        color = FAIL_COLOR
                        msg = "ACCESS DENIED"
                        self.log_access(status, name, 0.0)
                    
                    # Display result
                    cv2.putText(frame_display, msg, (20, 80),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.putText(frame_display, f"Similarity: {confirmed_similarity:.4f}", 
                              (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            # Draw info
            if confirmed_identity:
                cv2.putText(frame_display, f"User: {confirmed_identity}", (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
            
            cv2.putText(frame_display, f"Faces: {len(detections)}", 
                       (FRAME_WIDTH - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       TEXT_COLOR, 1)
            
            # FPS
            if DISPLAY_FPS:
                fps_counter += 1
                elapsed = (datetime.now() - fps_timer).total_seconds()
                if elapsed >= 1.0:
                    fps = fps_counter / elapsed
                    cv2.putText(frame_display, f"FPS: {fps:.1f}",
                              (FRAME_WIDTH - 200, FRAME_HEIGHT - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
                    fps_counter = 0
                    fps_timer = datetime.now()
            
            # Show
            cv2.imshow("Smart Door Lock", frame_display)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            frame_count += 1
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n[INFO] Access control system closed.")


def main():
    """Main entry point"""
    
    try:
        system = SmartDoorLock()
        system.run()
        
    except KeyboardInterrupt:
        print("\n[ABORT] Program interrupted by user!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
