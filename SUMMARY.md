# Face Recognition System - Summary

## Project Structure

```
FaceRecogition/
├── app.py                      # Flask main application
├── config.py                   # Configuration class
├── requirements.txt            # Dependencies
├── README.md                   # Full documentation
├── SUMMARY.md                  # This file
│
├── models/                     # Model wrappers
│   ├── __init__.py
│   ├── face_analyzer.py       # InsightFace singleton
│   └── model_utils.py         # Model utilities
│
├── recognition/                # Recognition pipeline
│   ├── __init__.py
│   ├── database.py            # Pickle/JSON storage
│   ├── face_recognizer.py     # Main recognizer class
│   ├── quality_check.py       # Image quality assessment
│   └── utils.py               # Draw, crop, normalize
│
├── training/                   # Training pipeline
│   ├── __init__.py
│   ├── data_preparation.py   # Collect, align, augment
│   ├── dataset.py            # PyTorch Dataset
│   ├── trainer.py            # Fine-tune ArcFace
│   └── utils.py              # Logging, metrics, viz
│
├── api/                        # Flask API
│   ├── __init__.py
│   ├── routes.py              # REST endpoints
│   ├── schemas.py             # Validation
│   └── error_handlers.py      # Error handling
│
├── scripts/                    # CLI utilities
│   ├── prepare_data.py       # Prepare dataset
│   ├── train.py              # Fine-tune model
│   ├── register_faces.py     # Register person (files/webcam)
│   ├── run_api.py            # Start API server
│   └── benchmark.py          # Performance benchmarks
│
├── tests/                      # Unit & integration tests
│   ├── test_recognition.py   # Database, quality, utils tests
│   ├── test_api.py           # API endpoint tests
│   └── conftest.py           # Pytest fixtures
│
├── static/                     # Web UI
│   ├── index.html            # Single-page app
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── data/                       # Data storage
│   ├── raw/                   # Original images by person
│   ├── processed/             # Aligned faces
│   ├── splits/                # Train/val manifests
│   └── database.pkl           # Embedding database
│
├── trained_models/             # Fine-tuned models
│   ├── checkpoints/
│   ├── finetuned/
│   └── logs/
│
├── uploads/                    # Temporary uploads
│   ├── temp/
│   └── faces/
│
├── logs/                       # Application logs
│
├── .env.example               # Environment variables template
├── Dockerfile                 # Docker image
├── docker-compose.yml         # Docker Compose
└── .dockerignore              # Docker ignore
```

## Features Implemented

### ✅ Core Recognition
- Face detection với InsightFace (buffalo_l model)
- Face alignment (5-point landmarks)
- Embedding extraction (512-D vectors)
- Cosine similarity matching
- File-based database (pickle/JSON)

### ✅ Quality Control
- Blur detection (Laplacian variance)
- Face size checking
- Pose estimation from landmarks
- Brightness/contrast assessment
- Symmetry check
- Configurable quality threshold

### ✅ Training Pipeline
- Data collection từ folder structure
- Face alignment & cropping
- Data augmentation (flip, rotate, color jitter, blur, noise)
- Train/val split
- PyTorch Dataset with on-the-fly augmentation
- Fine-tune ArcFace model
- ArcFace loss with angular margin
- Training logging & metrics
- Checkpoint saving
- Early stopping
- Learning rate scheduling
- ONNX export for deployment

### ✅ Flask REST API
- `/api/health` - Health check
- `/api/info` - System info & stats
- `/api/people` - List all people
- `/api/register` - Register new person (multiple images)
- `/api/recognize` - Detect & recognize faces
- `/api/detect` - Detect faces only
- `/api/person/<name>` (DELETE) - Remove person
- `/api/retrain` - Retrain model (placeholder)
- `/api/stats` - Detailed statistics
- Comprehensive error handling
- CORS enabled
- Request validation
- JSON responses

### ✅ CLI Scripts
- `scripts/prepare_data.py` - Full data prep
- `scripts/train.py` - Model fine-tuning
- `scripts/register_faces.py` - CLI registration (files/webcam)
- `scripts/run_api.py` - Start API server
- `scripts/benchmark.py` - Performance testing

### ✅ Web Interface
- Single-page app at `/`
- Register tab: upload multiple images, capture from webcam
- Recognize tab: upload image, adjust threshold, view results with bboxes
- People tab: list/delete registered people
- Responsive design

### ✅ Testing & Benchmark
- Unit tests cho database, quality, utils
- API integration tests
- Benchmark script cho inference speed, database search, memory
- Test fixtures with temp directories

### ✅ Deployment
- Dockerfile với all-in-one image
- Docker Compose configuration
- Volume mounts cho persistent data
- Health checks
- .dockerignore optimized

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start API server
python scripts/run_api.py

# 3. Open browser to http://localhost:5000 (web UI)
# Or use API directly:

# Register a person
curl -X POST http://localhost:5000/api/register \
  -F "name=John" \
  -F "images=@john1.jpg" \
  -F "images=@john2.jpg"

# Recognize faces
curl -X POST http://localhost:5000/api/recognize \
  -F "image=@test.jpg"
```

## Training Pipeline

```bash
# 1. Organize raw images
mkdir -p data/raw
# Put images in: data/raw/Person1/, data/raw/Person2/, ...

# 2. Prepare dataset (align, augment, split)
python scripts/prepare_data.py --source data/raw --output data/processed

# 3. Fine-tune model
python scripts/train.py \
  --train-manifest data/processed/train_manifest.csv \
  --val-manifest data/processed/val_manifest.csv \
  --epochs 50 \
  --batch-size 32

# Model saved to trained_models/finetuned/model.onnx
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build manually
docker build -t face-recognition .
docker run -p 5000:5000 -v $(pwd)/data:/app/data face-recognition
```

## Configuration

Edit `config.py` or set environment variables:
- `RECOGNITION_THRESHOLD`: Matching threshold (0-1)
- `DETECTION_THRESHOLD`: Face detection sensitivity
- `ENABLE_QUALITY_CHECK`: Quality filtering
- `TRAIN_BATCH_SIZE`, `TRAIN_EPOCHS`, etc.

## Performance

Typical performance on CPU:
- Detection: 10-30 FPS
- Recognition: 30-100 FPS
- Database search: <1ms (for <1000 people)

## Next Steps

1. **Collect quality data**: 5-10 clear, frontal face images per person
2. **Fine-tune model**: Use `scripts/train.py` with your dataset
3. **Adjust thresholds**: Tune `RECOGNITION_THRESHOLD` based on your accuracy requirements
4. **Production deployment**: Use Docker, add authentication, use vector DB for large datasets

## Notes

- Database lưu embeddings dạng pickle (fast) với JSON backup
- Mỗi person lưu tối đa `MAX_EMBEDDINGS_PER_PERSON` (mặc định 20)
- Webcam capture tạo ảnh quality check tự động
- Model buffalo_l tự động download lần đầu chạy (~300MB)
- Training support both ArcFace loss và triplet loss

## File Locations

- Database: `data/database.pkl`
- Logs: `logs/` hoặc `trained_models/logs/`
- Processed faces: `data/processed/`
- Trained models: `trained_models/finetuned/`

Made with ❤️ using InsightFace, Flask, PyTorch
