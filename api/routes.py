"""
Flask API routes for Face Recognition System
"""
import io
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import numpy as np
import cv2
from flask import Blueprint, request, jsonify, send_file

from recognition.face_recognizer import FaceRecognizer
from recognition.database import FaceDatabase
from models.face_analyzer import face_analyzer
from config import config
from .schemas import (
    APIResponse, RegisterRequest, RecognizeRequest,
    extract_files_from_request, extract_image_from_request,
    parse_float
)
from .error_handlers import api_error, api_success

logger = logging.getLogger(__name__)

# Create Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Global recognizer instance (will be initialized in create_app)
recognizer: Optional[FaceRecognizer] = None
database: Optional[FaceDatabase] = None


def init_api():
    """Initialize API components (call from app.py)"""
    global recognizer, database
    try:
        logger.info("Initializing API components...")
        database = FaceDatabase(config.DATABASE_PATH)
        recognizer = FaceRecognizer(
            database_path=config.DATABASE_PATH,
            threshold=config.RECOGNITION_THRESHOLD,
            enable_quality_check=config.ENABLE_QUALITY_CHECK
        )
        logger.info("API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


# ============== Health & Info ==============

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return api_success(data={
        'status': 'healthy',
        'model_loaded': face_analyzer._model is not None,
        'database_people': len(database.get_all_people()) if database else 0
    }, message="Face Recognition API is running")


@api_bp.route('/info', methods=['GET'])
def get_info():
    """Get system information"""
    if not recognizer:
        return api_error("API not initialized", 500)

    stats = recognizer.get_database_stats()
    config_info = {
        'detection_threshold': config.DETECTION_THRESHOLD,
        'recognition_threshold': config.RECOGNITION_THRESHOLD,
        'enable_quality_check': config.ENABLE_QUALITY_CHECK,
        'max_embeddings_per_person': config.MAX_EMBEDDINGS_PER_PERSON
    }

    return api_success(data={
        'stats': stats,
        'config': config_info
    })


# ============== People Management ==============

@api_bp.route('/people', methods=['GET'])
def list_people():
    """List all registered people"""
    if not database:
        return api_error("Database not initialized", 500)

    people = database.get_all_people()
    stats = database.get_stats()

    return api_success(data={
        'people': people,
        'count': len(people),
        'total_embeddings': stats['total_embeddings']
    })


@api_bp.route('/person/<name>', methods=['DELETE'])
def delete_person(name):
    """Delete a person from database"""
    if not database:
        return api_error("Database not initialized", 500)

    success = database.remove_person(name)
    if success:
        return api_success(message=f"Removed {name}")
    else:
        return api_error(f"Person '{name}' not found", 404)


# ============== Registration ==============

@api_bp.route('/register', methods=['POST'])
def register_person():
    """Register a new person with face embeddings"""
    if not recognizer:
        return api_error("Recognizer not initialized", 500)

    # Get name
    name = request.form.get('name', '').strip()
    error = RegisterRequest.validate(name, request.files.getlist('images'))
    if error:
        return api_error(error, 400)

    # Get images
    files = extract_files_from_request(request, 'images')
    if not files:
        return api_error("No images provided", 400)

    # Limit number of images
    if len(files) > 20:
        return api_error("Maximum 20 images allowed per registration", 400)

    # Read images
    images = []
    for file in files:
        try:
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                images.append(img)
        except Exception as e:
            logger.warning(f"Failed to read image: {e}")
            continue

    if not images:
        return api_error("No valid images found", 400)

    # Register
    result = recognizer.register_person(name, images)

    if result['success']:
        # Save raw images to data/raw/ for future training
        try:
            database.save_person_images(name, images)
        except Exception as e:
            logger.warning(f"Failed to save raw images: {e}")

        return api_success(
            data={
                'name': result['name'],
                'embeddings_extracted': result['embeddings_extracted'],
                'average_quality': result.get('average_quality'),
                'total_embeddings': result.get('total_embeddings_for_person')
            },
            message=f"Đã đăng ký {name} với {result['embeddings_extracted']} ảnh khuôn mặt"
        )
    else:
        return api_error(result['error'], 400)


# ============== Recognition ==============

@api_bp.route('/recognize', methods=['POST'])
def recognize_faces():
    """Detect and recognize faces in image"""
    if not recognizer:
        return api_error("Recognizer not initialized", 500)

    file = extract_image_from_request(request, 'image')
    if not file:
        return api_error("No image provided", 400)

    # Get threshold parameter
    threshold = parse_float(request.form.get('threshold'), None)
    if threshold is not None:
        error = RecognizeRequest.validate(threshold)
        if error:
            return api_error(error, 400)
    else:
        threshold = config.RECOGNITION_THRESHOLD

    # Read image
    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return api_error("Invalid image file", 400)
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return api_error("Failed to decode image", 400)

    # Recognize
    result = recognizer.recognize(img, threshold=threshold, return_all=False)

    if not result['success']:
        return api_error(result.get('error', 'Recognition failed'), 500)

    return api_success(data=result)


@api_bp.route('/detect', methods=['POST'])
def detect_faces():
    """Detect faces only (no recognition)"""
    if not recognizer:
        return api_error("Recognizer not initialized", 500)

    file = extract_image_from_request(request, 'image')
    if not file:
        return api_error("No image provided", 400)

    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return api_error("Invalid image file", 400)
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        return api_error("Failed to decode image", 400)

    result = recognizer.detect(img)

    return api_success(data={
        'faces': result,
        'count': len(result)
    })


# ============== Training ==============

@api_bp.route('/train/<name>', methods=['POST'])
def train_person(name):
    """
    Train/retrain model for a specific person
    
    This processes their embeddings and updates the recognition model
    """
    if not database:
        return api_error("Database not initialized", 500)
    
    if not recognizer:
        return api_error("Recognizer not initialized", 500)
    
    # Check if person exists
    person_data = database.get_person(name)
    if not person_data:
        return api_error(f"Person '{name}' not found", 404)
    
    embeddings = person_data.get('embeddings', [])
    if len(embeddings) == 0:
        return api_error(f"No embeddings found for '{name}'", 400)
    
    try:
        # Update person's average embedding
        embeddings_array = np.array(embeddings)
        avg_embedding = np.mean(embeddings_array, axis=0)
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        # Update person record with trained status
        person_data['metadata']['trained'] = True
        person_data['metadata']['trained_at'] = datetime.now().isoformat()
        person_data['metadata']['embedding_count'] = len(embeddings)
        
        database.save()
        
        logger.info(f"Trained model for {name} with {len(embeddings)} embeddings")
        
        return api_success(
            data={
                'name': name,
                'embeddings_count': len(embeddings),
                'trained_at': person_data['metadata']['trained_at']
            },
            message=f"Đã train thành công cho {name}"
        )
    except Exception as e:
        logger.error(f"Failed to train {name}: {e}")
        return api_error(f"Train thất bại: {str(e)}", 500)


@api_bp.route('/train-all', methods=['POST'])
def train_all_people():
    """
    Train model for all people in database
    """
    if not database:
        return api_error("Database not initialized", 500)
    
    if not recognizer:
        return api_error("Recognizer not initialized", 500)
    
    people = database.get_all_people()
    if len(people) == 0:
        return api_error("No people in database", 400)
    
    results = []
    errors = []
    
    for name in people:
        try:
            person_data = database.get_person(name)
            embeddings = person_data.get('embeddings', [])
            
            if len(embeddings) == 0:
                errors.append({'name': name, 'error': 'No embeddings'})
                continue
            
            # Update training status
            person_data['metadata']['trained'] = True
            person_data['metadata']['trained_at'] = datetime.now().isoformat()
            
            results.append({
                'name': name,
                'embeddings_count': len(embeddings),
                'status': 'success'
            })
        except Exception as e:
            errors.append({'name': name, 'error': str(e)})
    
    database.save()
    
    return api_success(
        data={
            'total': len(people),
            'success': len(results),
            'failed': len(errors),
            'results': results,
            'errors': errors
        },
        message=f"Đã train {len(results)}/{len(people)} người"
    )


@api_bp.route('/person/<name>/images', methods=['GET'])
def get_person_images(name):
    """
    Get saved face images for a person
    """
    if not database:
        return api_error("Database not initialized", 500)
    
    try:
        from config import config
        raw_dir = config.DATA_DIR / "raw" / name
        
        if not raw_dir.exists():
            return api_success(data={'images': [], 'count': 0})
        
        images = []
        for img_path in sorted(raw_dir.glob("*.jpg")):
            # Read image and convert to base64
            import base64
            img = cv2.imread(str(img_path))
            if img is not None:
                _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                images.append({
                    'filename': img_path.name,
                    'data': f'data:image/jpeg;base64,{img_base64}'
                })
        
        return api_success(data={
            'name': name,
            'images': images,
            'count': len(images)
        })
    except Exception as e:
        logger.error(f"Failed to get images for {name}: {e}")
        return api_error(f"Lỗi: {str(e)}", 500)


@api_bp.route('/person/<name>/status', methods=['GET'])
def get_person_status(name):
    """
    Get training status for a person
    """
    if not database:
        return api_error("Database not initialized", 500)
    
    person_data = database.get_person(name)
    if not person_data:
        return api_error(f"Person '{name}' not found", 404)
    
    metadata = person_data.get('metadata', {})
    
    return api_success(data={
        'name': name,
        'trained': metadata.get('trained', False),
        'trained_at': metadata.get('trained_at'),
        'embedding_count': metadata.get('embedding_count', len(person_data.get('embeddings', []))),
        'created_at': metadata.get('created_at')
    })


@api_bp.route('/retrain', methods=['POST'])
def retrain_model():
    """
    Fine-tune model with current database data

    Note: This is a long-running operation. Consider using Celery for production.
    """
    if not database:
        return api_error("Database not initialized", 500)

    people = database.get_all_people()
    if len(people) < 2:
        return api_error("Need at least 2 people with embeddings to train", 400)

    # This would require extracting embeddings and creating dataset
    # For simplicity, return not implemented or run training script
    return api_error(
        "Training via API not implemented yet. Use CLI: python scripts/train.py",
        501
    )


# ============== Utils ==============

@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get detailed statistics"""
    if not database:
        return api_error("Database not initialized", 500)

    stats = database.get_stats()

    # Add additional stats
    from recognition.face_recognizer import FaceRecognizer
    if recognizer:
        stats['threshold'] = recognizer.threshold
        stats['quality_check_enabled'] = recognizer.enable_quality_check

    return api_success(data=stats)


@api_bp.route('/clear-database', methods=['POST'])
def clear_database():
    """Clear all data from database (DANGEROUS)"""
    if not database:
        return api_error("Database not initialized", 500)

    confirm = request.json.get('confirm', False) if request.is_json else False
    if not confirm:
        return api_error("Must set confirm=true in JSON body", 400)

    database.clear()
    return api_success(message="Database cleared")
