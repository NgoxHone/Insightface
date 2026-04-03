# Face Recognition System with InsightFace & Flask API

Hệ thống nhận diện khuôn mặt hoàn chỉnh với khả năng training, fine-tuning và REST API.

## Features

- **Face Detection & Recognition**: Sử dụng InsightFace (ArcFace)
- **Fine-tuning**: Train model với dữ liệu của bạn
- **REST API**: Đầy đủ endpoints cho đăng ký, nhận diện, quản lý
- **Quality Checking**: Lọc ảnh chất lượng thấp tự động
- **Face Alignment**: căn chỉnh khuôn mặt trước khi extract embedding
- **File-based Database**: Lưu embeddings dạng JSON/PKL
- **Training Pipeline**: Thu thập data, preprocessing, augmentation, training

## Project Structure

```
FaceRecogition/
├── app.py                      # Flask main application
├── config.py                   # Configuration
├── requirements.txt
├── README.md
│
├── models/                     # InsightFace model wrappers
│   ├── __init__.py
│   ├── face_analyzer.py
│   └── model_utils.py
│
├── recognition/                # Recognition logic
│   ├── __init__.py
│   ├── database.py             # Embedding storage
│   ├── face_recognizer.py      # Main recognizer
│   ├── quality_check.py        # Quality assessment
│   └── utils.py
│
├── training/                   # Training pipeline
│   ├── __init__.py
│   ├── data_collection.py
│   ├── data_preparation.py
│   ├── dataset.py
│   ├── trainer.py
│   └── utils.py
│
├── api/                       # Flask routes
│   ├── __init__.py
│   ├── routes.py
│   ├── schemas.py
│   └── error_handlers.py
│
├── scripts/                   # CLI utilities
│   ├── prepare_data.py
│   ├── train.py
│   ├── register_faces.py
│   └── run_api.py
│
├── data/
│   ├── raw/                   # Original images (organized by person)
│   │   ├── John/
│   │   │   ├── img1.jpg
│   │   │   └── img2.jpg
│   │   └── Alice/
│   ├── processed/             # Aligned faces
│   └── splits/                # Train/val splits
│
├── trained_models/            # Fine-tuned models
│   ├── checkpoints/
│   ├── finetuned/
│   └── logs/
│
├── uploads/                   # Temporary uploads
│   ├── temp/
│   └── faces/
│
└── tests/                     # Unit tests
    ├── test_recognition.py
    └── test_api.py
```

## Installation

```bash
# Clone or navigate to project directory
cd FaceRecogition

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# InsightFace will auto-download models on first run (~300MB)
```

## Quick Start

### 1. Start the Flask API

```bash
python scripts/run_api.py
# or
python app.py
```

API sẽ chạy tại `http://localhost:5000`

### 2. Register a new person

```bash
curl -X POST http://localhost:5000/api/register \
  -F "name=John Doe" \
  -F "images=@john1.jpg" \
  -F "images=@john2.jpg" \
  -F "images=@john3.jpg"
```

Response:
```json
{
  "success": true,
  "message": "Registered John Doe with 3 face embeddings",
  "face_count": 3
}
```

### 3. Recognize faces

```bash
curl -X POST http://localhost:5000/api/recognize \
  -F "image=@test.jpg" \
  -F "threshold=0.6"
```

Response:
```json
{
  "success": true,
  "faces": [
    {
      "bbox": [100, 50, 200, 150],
      "name": "John Doe",
      "confidence": 0.89,
      "landmarks": [[...], [...], ...]
    }
  ]
}
```

### 4. Only detect faces

```bash
curl -X POST http://localhost:5000/api/detect \
  -F "image=@test.jpg"
```

### 5. List all registered people

```bash
curl http://localhost:5000/api/people
```

### 6. Delete a person

```bash
curl -X DELETE http://localhost:5000/api/person/John%20Doe
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/people` | GET | List all registered people |
| `/api/register` | POST | Register new person with face embeddings |
| `/api/recognize` | POST | Detect and recognize faces in image |
| `/api/detect` | POST | Detect faces only (no recognition) |
| `/api/person/<name>` | DELETE | Remove person from database |
| `/api/retrain` | POST | Fine-tune model with current database data |

### Request/Response Schemas

#### Register Request
- `name` (string, required): Person's name
- `images` (files, required): Multiple face images

#### Recognize Request
- `image` (file, required): Image to recognize
- `threshold` (float, optional): Similarity threshold (default: 0.6)

#### Response Format
```json
{
  "success": true,
  "data": { ... },
  "error": null
}
```

## Training & Fine-tuning

### Prepare Dataset

1. Organize raw images:
```
data/raw/
├── John/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Alice/
└── Bob/
```

2. Run data preparation:
```bash
python scripts/prepare_data.py --source data/raw --output data/processed
```

This will:
- Detect faces in all images
- Align faces using 5-point landmarks
- Apply augmentations
- Generate train/val splits

### Fine-tune Model

```bash
python scripts/train.py \
  --data data/processed \
  --epochs 50 \
  --batch-size 32 \
  --lr 0.001 \
  --output trained_models/finetuned
```

Options:
- `--resume`: Resume from checkpoint
- `--model`: Pretrained model path (default: buffalo_l)
- `--arch`: Model architecture (arcface, adaface)

The script will:
- Load pre-trained ArcFace
- Fine-tune on your dataset
- Save best model to `trained_models/best.onnx`
- Log metrics to `trained_models/logs/`

## Configuration

Edit `config.py` to customize:

```python
# Detection threshold (lower = more faces, more false positives)
DETECTION_THRESHOLD = 0.5

# Recognition threshold (higher = more strict matching)
RECOGNITION_THRESHOLD = 0.6

# Enable/disable features
ENABLE_ALIGNMENT = True
ENABLE_QUALITY_CHECK = True

# Training parameters
TRAIN_BATCH_SIZE = 32
TRAIN_EPOCHS = 50
TRAIN_LEARNING_RATE = 0.001
```

## Command Line Tools

### Register faces from folder
```bash
python scripts/register_faces.py --name "John" --images /path/to/john/*.jpg
```

### Register from webcam
```bash
python scripts/register_faces.py --name "Alice" --webcam --count 50
```

### Run API server
```bash
python scripts/run_api.py --host 0.0.0.0 --port 5000 --debug
```

## Web Interface (Optional)

A simple web UI is available at `http://localhost:5000/`:

- Upload images to register new people
- Upload images to recognize faces
- View all registered people
- Real-time webcam recognition (via web UI)

## Accuracy Improvement Tips

1. **Quality Images**: Use high-resolution, well-lit, frontal face images for registration
2. **Multiple Samples**: Register 5-10 images per person with variations (lighting, expression)
3. **Face Alignment**: System automatically aligns faces for consistency
4. **Fine-tuning**: Train on your specific dataset for better domain adaptation
5. **Threshold Tuning**: Adjust `RECOGNITION_THRESHOLD` based on your use case:
   - Lower (0.4-0.5): More permissive, fewer false negatives
   - Higher (0.7-0.8): More strict, fewer false positives
6. **Quality Filtering**: Enable quality checks to reject blurry/low-quality faces

## Performance

- **Detection Speed**: ~10-30 FPS on CPU, ~50-100 FPS on GPU
- **Recognition Speed**: ~50-100 FPS (batch size 1)
- **Memory**: ~500MB for model + database

Optimizations:
- Batch processing enabled by default
- Model caching
- Embedding caching for registered people

## Troubleshooting

### Model not found
InsightFace will auto-download on first run. If it fails:
```bash
python -c "import insightface; insightface.app.FaceAnalysis(name='buffalo_l')"
```

### CUDA/GPU support
Install CUDA version of onnxruntime:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### Permission errors
Ensure upload folders exist and are writable:
```bash
mkdir -p uploads/temp uploads/faces data/processed
```

## Testing

Run tests:
```bash
pytest tests/ -v
```

Run benchmark:
```bash
python scripts/benchmark.py --test-data data/val/ --model trained_models/best.onnx
```

## Development

Add new features:
- API endpoints: edit `api/routes.py`
- Recognition logic: edit `recognition/face_recognizer.py`
- Training: edit `training/trainer.py`

## License

MIT License - Feel free to use for your projects!

## Credits

- [InsightFace](https://github.com/deepinsight/insightface) - State-of-the-art face recognition
- Built with Flask, OpenCV, PyTorch


fe: cd frontend && npm run dev
be: python3 scripts/run_api.py --host 0.0.0.0 --port 5001 --debug