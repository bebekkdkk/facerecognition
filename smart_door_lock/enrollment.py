"""
Enrollment Script - Face Enrollment Module
- Capture wajah pengguna baru
- Extract embedding
- Simpan ke database
- Optimized untuk Raspberry Pi
"""

import cv2
import sys
import os
import gc
from datetime import datetime

# Add parent directory ke path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, 
    ENROLLMENT_SAMPLES, ENROLLMENT_SAMPLE_INTERVAL,
    TEXT_COLOR, FAIL_COLOR, DISPLAY_FPS,
    CAMERA_WARMUP_FRAMES, MEMORY_CLEANUP_INTERVAL,
    IS_RASPBERRY_PI
)
from modules import ImagePreprocessor, FaceDetector, FaceEmbedder, FaceDatabase


def main():
    """Main enrollment process"""
    
    # Get username
    print("\n" + "="*50)
    print("SMART DOOR LOCK - ENROLLMENT SYSTEM")
    print("="*50)
    
    name = input("\n[INPUT] Enter your name: ").strip()
    if not name:
        print("[ERROR] Name cannot be empty!")
        return False
    
    print(f"\n[INFO] Starting enrollment for: {name}")
    print(f"[INFO] Preparing to capture {ENROLLMENT_SAMPLES} face samples...")
    
    # Initialize camera dengan validation
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    # Check camera accessibility
    if not cap.isOpened():
        print(f"[ERROR] Camera {CAMERA_INDEX} is not accessible!")
        print("[ERROR] Please check camera connection and permissions.")
        return False
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer untuk minimize latency
    
    # Warmup camera - capture beberapa frames tanpa processing
    print(f"\n[ACTION] Warming up camera ({CAMERA_WARMUP_FRAMES} frames)...")
    for i in range(CAMERA_WARMUP_FRAMES):
        ret, _ = cap.read()
        if not ret:
            print("[ERROR] Camera warmup failed!")
            cap.release()
            return False
    
    # Initialize components
    try:
        preprocessor = ImagePreprocessor()
        detector = FaceDetector()  # Use local Haar Cascade
        embedder = FaceEmbedder()
        database = FaceDatabase()
        database.create_table()
        print("[INFO] All components initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize components: {e}")
        cap.release()
        return False
    
    # Enrollment variables
    captured_embeddings = []
    frame_count = 0
    sample_count = 0
    fps_counter = 0
    fps_timer = datetime.now()
    
    print("\n[ACTION] Position your face in the center. Press 'q' to quit.\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read from camera!")
                break
            
            # Preprocess frame
            frame = preprocessor.resize_frame(frame)
            
            # Detect faces
            detections = detector.detect(frame)
            
            # Draw detections
            frame_display = detector.draw_detections(
                frame, detections, 
                color=TEXT_COLOR, 
                show_confidence=True
            )
            
            # Process detections
            for detection in detections:
                x, y, w, h, conf = detection
                
                if conf < 0.5:
                    continue
                
                # Crop wajah
                face_region = frame[y:y+h, x:x+w]
                
                # Preprocess untuk embedding
                face_prepared = preprocessor.prepare_for_embedding(face_region)
                
                # Extract embedding setiap ENROLLMENT_SAMPLE_INTERVAL frame
                if frame_count % ENROLLMENT_SAMPLE_INTERVAL == 0:
                    embedding = embedder.extract(face_prepared)
                    captured_embeddings.append(embedding)
                    sample_count += 1
                    
                    print(f"\r[CAPTURE] Samples collected: {sample_count}/{ENROLLMENT_SAMPLES}", end="", flush=True)
                    
                    # If sudah cukup, break
                    if sample_count >= ENROLLMENT_SAMPLES:
                        print()
                        break
            
            # Draw info
            info_text = f"Captured: {sample_count}/{ENROLLMENT_SAMPLES}"
            cv2.putText(frame_display, info_text, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)
            
            # FPS display only if not on RPi
            if DISPLAY_FPS and not IS_RASPBERRY_PI:
                fps_counter += 1
                elapsed = (datetime.now() - fps_timer).total_seconds()
                if elapsed >= 1.0:
                    fps = fps_counter / elapsed
                    cv2.putText(frame_display, f"FPS: {fps:.1f}", (20, FRAME_HEIGHT - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
                    fps_counter = 0
                    fps_timer = datetime.now()
            
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
            
            # Memory cleanup pada interval tertentu (untuk RPi)
            if frame_count % MEMORY_CLEANUP_INTERVAL == 0:
                gc.collect()
            
            # Check jika sudah capture cukup
            if sample_count >= ENROLLMENT_SAMPLES:
                break
    
    except Exception as e:
        print(f"\n[ERROR] Error during enrollment: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        gc.collect()
    
    # Check if enrollment successful
    if len(captured_embeddings) < ENROLLMENT_SAMPLES:
        print(f"\n[ERROR] Failed to capture {ENROLLMENT_SAMPLES} samples!")
        print(f"[INFO] Only captured {len(captured_embeddings)} samples.")
        return False
    
    # Save ke database
    print("\n[PROCESSING] Saving embeddings to database...")
    
    success = database.add_enrollment(name, captured_embeddings)
    
    if success:
        # Print completion info
        stats = database.get_stats()
        print("\n" + "="*50)
        print(f"[SUCCESS] Enrollment completed for: {name}")
        print(f"[INFO] Total embeddings saved: {len(captured_embeddings)}")
        print(f"[INFO] Database stats:")
        print(f"       - Total users: {stats.get('total_users', 0)}")
        print(f"       - Total embeddings: {stats.get('total_embeddings', 0)}")
        if IS_RASPBERRY_PI:
            print("[INFO] Enrollment optimized for Raspberry Pi")
        print("="*50 + "\n")
        return True
    else:
        print(f"\n[ERROR] Failed to save embeddings to database!")
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
