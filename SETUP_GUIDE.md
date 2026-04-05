# Smart Door Lock - Face Recognition + Anti-Spoofing System
## TensorFlow Lite + Raspberry Pi 3 Optimized

Sistem face recognition dengan anti-spoofing detection yang FULL kompatibel dengan Raspberry Pi 3 dan Python 3.9.2.

### ✅ Features

- **Face Detection**: OpenCV Haar Cascade (reliable, lightweight)
- **Face Embedding**: MobileFaceNet.tflite (TensorFlow Lite)
- **Anti-Spoofing**: FaceAntiSpoofing.tflite dengan Laplacian blur detection
- **Multi-Pose Enrollment**: Capture 5 pose wajah (depan, kiri, kanan, atas, bawah)
- **Recognition**: Cosine similarity matching dengan threshold 0.7
- **Real-time Processing**: Optimized untuk Raspberry Pi 3
- **Zero ONNX/Heavy Dependencies**: Pure TensorFlow Lite + OpenCV

### 📁 Project Structure

```
smart_door_lock/
├── main.py                      # Main application - Real-time face recognition
├── enrollment.py                # Multi-pose enrollment system (5 poses)
├── config.py                    # Configuration (optimized untuk RPi)
├── models/
│   ├── MobileFaceNet.tflite      # Embedding model
│   ├── FaceAntiSpoofing.tflite   # Anti-spoofing model
│   └── haarcascade_frontalface_default.xml
├── data/
│   ├── access_log.json           # Access logs
│   └── face_database.db          # User embeddings database
└── modules/
    ├── __init__.py
    ├── anti_spoofing.py          # Anti-spoofing detection logic
    ├── embedder.py               # MobileFaceNet embedding extraction
    ├── recognition.py            # Recognition pipeline
    ├── face_detector.py          # Haar Cascade face detection
    ├── database.py               # Database operations
    ├── preprocessing.py          # Image preprocessing
    └── tracker.py                # Face tracking (optional)
```

### 🚀 Installation

#### For Desktop (Linux/Windows/Mac)

```bash
# Clone or navigate to project
cd d:\cobaskripsi

# Install dependencies
pip install -r requirements.txt

# Run application
python main.py
```

#### For Raspberry Pi 3

```bash
# SSH ke Raspberry Pi
ssh pi@raspberrypi.local

# Update system
sudo apt-get update
sudo apt-get upgrade

# Install Python 3.9.2 (if not installed)
sudo apt-get install python3.9 python3.9-dev

# Navigate ke project
cd /path/to/smart_door_lock

# Install requirements (RPi optimized)
python3.9 -m pip install -r requirements-rpi.txt

# Run application
python3.9 main.py
```

### 📋 Usage

#### 1. Enrollment (Capture 5 Poses)

```bash
python enrollment.py
```

Flow:
1. Masukkan nama/username
2. System akan meminta 5 pose berbeda:
   - **Depan** (front) - langsung ke kamera
   - **Kiri** (left) - miringkan kepala ke kiri
   - **Kanan** (right) - miringkan kepala ke kanan
   - **Atas** (up) - angkat kepala ke atas
   - **Bawah** (down) - turunkan kepala ke bawah
3. Untuk setiap pose, tunggu capture automatic (~3 detik)
4. System akan average 5 embeddings menjadi 1 vector
5. Simpan ke database

#### 2. Face Recognition (Real-time)

```bash
python main.py
```

Flow:
1. Open video stream dari camera
2. Detect wajah setiap frame
3. Jalankan anti-spoofing check terlebih dahulu
4. Jika REAL → extract embedding
5. Compare dengan database menggunakan cosine similarity
6. Display hasil: status spoofing + nama user + similarity score
7. Jika similarity > 0.7 → MATCH

**Keyboard Controls:**
- `q` - Quit aplikasi
- `e` - Start enrollment (dari menu)

### 🔧 Configuration

Edit `config.py` untuk customize:

```python
# Camera settings
FRAME_WIDTH = 480      # Lower untuk RPi
FRAME_HEIGHT = 360
FPS = 20              # Lower FPS = less CPU usage

# Anti-spoofing
ANTI_SPOOF_THRESHOLD = 0.2    # Score < 0.2 → REAL
ANTI_SPOOF_LAPLACE_THRESHOLD = 50  # Blur detection

# Recognition
RECOGNITION_THRESHOLD = 0.7   # Cosine similarity threshold
```

### 🧪 Testing Anti-Spoofing

Model anti-spoofing menggunakan:
- **Laplacian blur detection** (variance < 50 → reject blur)
- **FaceAntiSpoofing.tflite model** (score < 0.2 → REAL)

Test dengan:
- Wajah asli → REAL ✓
- Screenshot/printed photo → FAKE ✗
- Video replay → FAKE ✗
- Mirror reflection → FAKE ✗

### 📊 Model Information

#### MobileFaceNet.tflite
- **Input**: 112x112x3 RGB normalized
- **Output**: 128-dim embedding
- **Normalization**: (pixel - 0.5) / 0.5
- **Inference time**: ~20ms pada RPi 3

#### FaceAntiSpoofing.tflite
- **Input**: 256x256 normalized
- **Output**: Spoofing probability scores
- **Logic**: Laplacian blur + model score
- **Threshold**: 0.2 (tunable di config.py)

#### Haar Cascade
- **File**: haarcascade_frontalface_default.xml
- **Detection**: Fast, reliable, RPi-friendly
- **Tunable**: scaleFactor=1.3, minNeighbors=5

### 🎯 Algorithm Details

#### Anti-Spoofing Flow
1. Resize frame ke 256x256
2. Normalize: pixel / 255.0
3. Hitung Laplacian variance dengan kernel 3x3
4. Jika variance < 50 → REJECT (blur image)
5. Jika lolos → jalankan FaceAntiSpoofing.tflite
6. Score = sum(abs(pred[i]) * mask[i])
7. Jika score < 0.2 → **REAL** ✓
   Jika score >= 0.2 → **FAKE** ✗

#### Enrollment Strategy
- Capture 5 pose berbeda untuk coverage yang lebih baik
- Extract embedding dari setiap pose
- Average embeddings: final_vector = mean(emb1, emb2, ..., emb5)
- L2 normalize hasil akhir
- Simpan 1 vector per user (128-dim)

#### Recognition Matching
1. Anti-spoofing check → REAL/FAKE classification
2. Jika FAKE → reject langsung
3. Jika REAL → extract embedding
4. Hitung cosine similarity dengan semua user embeddings:
   ```
   similarity = dot(emb_capture, emb_database) / (norm(emb_capture) * norm(emb_database))
   ```
5. Ambil max similarity
6. Jika max_sim >= 0.7 → MATCH (display nama user)
   Jika max_sim < 0.7 → UNKNOWN (display "UNKNOWN")

### 📈 Performance

#### Raspberry Pi 3
- **Face Detection**: ~30ms
- **Anti-spoofing**: ~50ms
- **Embedding Extraction**: ~20ms
- **Total latency**: ~100ms per frame
- **FPS**: ~10-15 FPS dengan real-time display

#### Desktop (Intel i5)
- **Total latency**: ~30-50ms
- **FPS**: ~20-30 FPS

### 🔐 Security Notes

- Anti-spoofing menggunakan model berbasis pembelajaran (lebih aman dari rule-based)
- Laplacian blur detection mendeteksi screenshot/printed photo
- 5-pose enrollment meningkatkan robustness
- Cosine similarity threshold 0.7 = balanced false positives/negatives

### ⚠️ Raspberry Pi 3 Optimization

```python
IS_RASPBERRY_PI = True  # Auto-detect

# Automatically optimized untuk RPi:
- FRAME_WIDTH = 480x360 (vs 640x480 desktop)
- FPS = 20 (vs 30 desktop)
- Memory cleanup every 100 frames
- No GPU acceleration needed (CPU only)
- No threading (sequential processing)
```

### 🐛 Troubleshooting

#### Camera Not Found
```bash
# Check kamera
ls /dev/video*

# Grant permissions
sudo usermod -a -G video $USER
```

#### tflite-runtime Not Found
```bash
# Install untuk Python 3.9
python3.9 -m pip install tflite-runtime

# Or untuk development
pip install tensorflow==2.13.0
```

#### Models Not Found
Ensure models exist di:
```
smart_door_lock/models/
├── MobileFaceNet.tflite
├── FaceAntiSpoofing.tflite
└── haarcascade_frontalface_default.xml
```

#### Low FPS di Raspberry Pi
- Reduce FRAME_WIDTH/FRAME_HEIGHT di config.py
- Disable display: `visualize=False`
- Close other processes

### 📝 Database Structure

Face embeddings disimpan di `data/face_database.db` menggunakan LanceDB:

```
User Table:
- id: unique identifier
- name: username
- embedding: 128-dim vector (L2 normalized)
- created_at: timestamp
- last_seen: last recognition timestamp
```

### 🔄 Workflow

```
enrollment.py
    ↓
[Capture 5 poses] → [Extract embeddings] → [Average] → [Store in DB]
                                                           ↓
                                                    main.py
                                                    (Real-time)
                                                           ↓
[Detect face] → [Anti-spoof] → [Extract emb] → [Compare] → [Display result]
                                                    ↓
                                            LanceDB (cosine similarity)
```

### 📚 Dependencies

- **cv2** (OpenCV): Face detection & image processing
- **numpy**: Matrix operations
- **tflite-runtime**: TensorFlow Lite inference (lightweight)
- **lancedb**: Vector database untuk similarity search
- **pyarrow**: Data serialization

**Zero dependencies pada:**
- ❌ onnxruntime (too heavy for RPi)
- ❌ TensorFlow full framework (500MB+)
- ❌ pytorch/torch
- ❌ scikit-learn heavy versions

### 📄 License

Optimized untuk Raspberry Pi 3 + Python 3.9.2

### 💡 Tips untuk Accuracy

1. **Enrollment**: Position wajah centered, good lighting, clear face
2. **Recognition**: Same lighting conditions, similar angles
3. **Anti-spoofing**: Natural face movements, not too close to camera
4. **Database**: Add more users untuk testing robustness

### 🚦 Next Steps

1. Deploy ke Raspberry Pi
2. Setup database dengan users
3. Integrate dengan access control system
4. Monitor performa dan adjust thresholds
5. Add logging untuk audit trail

---

**Last Updated**: 2026-04-05
**Compatibility**: Python 3.9.2+, Raspberry Pi 3, Linux/Windows/Mac
**Main Models**: MobileFaceNet.tflite, FaceAntiSpoofing.tflite
