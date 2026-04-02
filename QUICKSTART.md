# Face Recognition System - Quick Start Guide

## Installation Complete! 🎉

Your face recognition system is ready to use.

## Environment

- Python 3.10 (venv activated)
- Dependencies installed: insightface, flask, torch, opencv-python-headless, numpy 1.26.4
- All modules verified and compiled

## What Was Built

**30+ files** organized into:

```
FaceRecogition/
├── Core: models/, recognition/ (database, recognizer, quality)
├── Training: training/ (data prep, dataset, trainer)
├── API: app.py, api/ (routes, schemas, errors)
├── Scripts: prepare_data, train, register_faces, run_api, benchmark
├── UI: static/index.html + CSS/JS
└── Deploy: Dockerfile, docker-compose.yml
```

## Quick Test

### 1. Start the API server

```bash
cd /Users/mac/Desktop/CodeTime/Python/FaceRecogition
source venv/bin/activate
python3 scripts/run_api.py
```

Server will start at **http://localhost:5000**

### 2. Test endpoints

```bash
# Health check
curl http://localhost:5000/api/health

# List people
curl http://localhost:5000/api/people

# Register a person (with face1.jpg already in project root)
curl -X POST http://localhost:5000/api/register \
  -F "name=Test User" \
  -F "images=@face1.jpg"

# Recognize faces
curl -X POST http://localhost:5000/api/recognize \
  -F "image=@face1.jpg"
```

### 3. Web Interface

Open browser to **http://localhost:5000/**

- **Recognize tab**: Upload image → Detect & recognize faces
- **Register tab**: Enter name + upload multiple face images
- **People tab**: View/delete registered people

## Training Your Own Model

```bash
# 1. Prepare your data
# Put face images in: data/raw/PersonName/*.jpg (5+ images per person)

python scripts/prepare_data.py --source data/raw --output data/processed

# 2. Fine-tune ArcFace
python scripts/train.py \
  --train-manifest data/processed/train_manifest.csv \
  --val-manifest data/processed/val_manifest.csv \
  --epochs 50 \
  --batch-size 32

# Model saved to: trained_models/finetuned/model.onnx
```

## Configuration

Edit `config.py` or set environment variables:

```python
RECOGNITION_THRESHOLD = 0.6     # Matching threshold (higher = stricter)
DETECTION_THRESHOLD = 0.5       # Face detection sensitivity
ENABLE_QUALITY_CHECK = True    # Filter blurry faces
MAX_EMBEDDINGS_PER_PERSON = 20 # Store up to 20 embeddings per person
```

## Webcam Registration

```bash
python scripts/register_faces.py --name "Alice" --webcam --count 50
```

- Press SPACE to capture faces
- ESC to quit
- Face crops saved to `uploads/faces/`

## Docker Deployment

```bash
docker-compose up -d
# API available at http://localhost:5000
```

## Troubleshooting

**Import errors?** Make sure you're in venv:
```bash
source venv/bin/activate
```

**Model download slow?** The buffalo_l model (~281MB) downloads on first run to `models/buffalo_l/`. You can pre-download:

```bash
python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l')"
```

**Port already in use?** Change port:
```bash
python scripts/run_api.py --port 8080
```

**Out of memory?** Reduce batch size in config:
```python
TRAIN_BATCH_SIZE = 16
```

## File Locations

- **Database**: `data/database.pkl` (auto-created)
- **Uploads**: `uploads/faces/`
- **Logs**: `logs/face_recognition.log`
- **Trained models**: `trained_models/finetuned/`

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/info` | GET | System info & stats |
| `/api/people` | GET | List all people |
| `/api/register` | POST | Register new person |
| `/api/recognize` | POST | Detect & recognize |
| `/api/detect` | POST | Detect faces only |
| `/api/person/<name>` | DELETE | Delete person |

## Accuracy Tips

1. **Use quality images**: Well-lit, frontal faces
2. **Multiple samples**: 5-10 images per person with variations
3. **Fine-tune**: Train on your specific dataset
4. **Adjust threshold**: Lower (0.4-0.5) for permissive matching, higher (0.7-0.8) for strict
5. **Quality check enabled**: Rejects blurry/low-quality faces automatically

---

**Need help?** See full README.md for detailed documentation.

Enjoy your face recognition system! 🚀
