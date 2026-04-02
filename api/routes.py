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
    """List all people from data/raw directory"""
    try:
        from config import config
        raw_dir = config.DATA_DIR / "raw"
        
        if not raw_dir.exists():
            return api_success(data={
                'people': [],
                'count': 0,
                'total_embeddings': 0,
                'total_images': 0
            })
        
        people_data = []
        total_images = 0
        total_embeddings = 0
        
        # Scan data/raw directory
        for person_dir in sorted(raw_dir.iterdir()):
            if not person_dir.is_dir() or person_dir.name.startswith('.'):
                continue
            
            # Count images
            images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))
            image_count = len(images)
            total_images += image_count
            
            # Check if person is in database
            person_in_db = database.get_person(person_dir.name) if database else None
            embeddings_count = len(person_in_db.get('embeddings', [])) if person_in_db else 0
            total_embeddings += embeddings_count
            
            is_trained = person_in_db is not None and embeddings_count > 0
            trained_at = person_in_db.get('metadata', {}).get('trained_at') if person_in_db else None
            
            people_data.append({
                'name': person_dir.name,
                'image_count': image_count,
                'embeddings_count': embeddings_count,
                'is_trained': is_trained,
                'trained_at': trained_at,
                'has_database': person_in_db is not None
            })
        
        return api_success(data={
            'people': people_data,
            'count': len(people_data),
            'total_embeddings': total_embeddings,
            'total_images': total_images
        })
    except Exception as e:
        logger.error(f"Failed to list people: {e}")
        return api_error(f"Lỗi: {str(e)}", 500)


@api_bp.route('/person/<name>', methods=['GET'])
def get_person_detail(name):
    """Get detailed info about a person including images"""
    try:
        from config import config
        import base64
        
        raw_dir = config.DATA_DIR / "raw" / name
        if not raw_dir.exists():
            return api_error(f"Person '{name}' not found in data/raw", 404)
        
        # Get images
        images = []
        for img_path in sorted(raw_dir.glob("*.*")):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        img_base64 = base64.b64encode(buffer).decode('utf-8')
                        images.append({
                            'filename': img_path.name,
                            'path': str(img_path.relative_to(config.DATA_DIR / "raw")),
                            'data': f'data:image/jpeg;base64,{img_base64}',
                            'size': img_path.stat().st_size
                        })
                except Exception as e:
                    logger.warning(f"Failed to read {img_path}: {e}")
        
        # Get database info
        person_in_db = database.get_person(name) if database else None
        embeddings_count = len(person_in_db.get('embeddings', [])) if person_in_db else 0
        is_trained = person_in_db is not None and embeddings_count > 0
        trained_at = person_in_db.get('metadata', {}).get('trained_at') if person_in_db else None
        
        return api_success(data={
            'name': name,
            'images': images,
            'image_count': len(images),
            'embeddings_count': embeddings_count,
            'is_trained': is_trained,
            'trained_at': trained_at,
            'has_database': person_in_db is not None
        })
    except Exception as e:
        logger.error(f"Failed to get person {name}: {e}")
        return api_error(f"Lỗi: {str(e)}", 500)


@api_bp.route('/person/<name>/images', methods=['POST'])
def upload_person_images(name):
    """
    Upload new images for a person to data/raw/<name>/
    """
    try:
        from config import config
        
        # Get uploaded files
        files = request.files.getlist('images')
        if not files or len(files) == 0:
            return api_error("No images provided", 400)
        
        # Create directory if not exists
        raw_dir = config.DATA_DIR / "raw" / name
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Get existing count
        existing = list(raw_dir.glob("*.*"))
        start_idx = len(existing) + 1
        
        saved_count = 0
        for i, file in enumerate(files):
            if file and file.filename:
                # Generate filename
                ext = Path(file.filename).suffix or '.jpg'
                filename = f"{start_idx + i:04d}{ext}"
                filepath = raw_dir / filename
                
                # Save file
                file.save(str(filepath))
                saved_count += 1
        
        logger.info(f"Saved {saved_count} images for {name} to {raw_dir}")
        
        return api_success(
            data={
                'name': name,
                'saved_count': saved_count,
                'total_images': len(list(raw_dir.glob("*.*")))
            },
            message=f"Đã thêm {saved_count} ảnh cho {name}"
        )
    except Exception as e:
        logger.error(f"Failed to upload images for {name}: {e}")
        return api_error(f"Lỗi: {str(e)}", 500)


@api_bp.route('/person/<name>/image/<filename>', methods=['DELETE'])
def delete_person_image(name, filename):
    """
    Delete a specific image from data/raw/<name>/
    """
    try:
        from config import config
        
        raw_dir = config.DATA_DIR / "raw" / name
        filepath = raw_dir / filename
        
        if not filepath.exists():
            return api_error(f"Image '{filename}' not found", 404)
        
        filepath.unlink()
        logger.info(f"Deleted {filepath} for {name}")
        
        return api_success(message=f"Đã xóa ảnh {filename}")
    except Exception as e:
        logger.error(f"Failed to delete image: {e}")
        return api_error(f"Lỗi: {str(e)}", 500)


@api_bp.route('/person/<name>', methods=['DELETE'])
def delete_person(name):
    """Delete a person from database and data/raw"""
    if not database:
        return api_error("Database not initialized", 500)

    try:
        from config import config
        
        # Remove from database
        db_success = database.remove_person(name)
        
        # Remove raw images
        raw_dir = config.DATA_DIR / "raw" / name
        if raw_dir.exists():
            import shutil
            shutil.rmtree(raw_dir)
            logger.info(f"Removed raw images for {name}")
        
        if db_success or raw_dir.exists() is False:
            return api_success(message=f"Đã xóa {name}")
        else:
            return api_error(f"Person '{name}' not found", 404)
    except Exception as e:
        logger.error(f"Failed to delete person {name}: {e}")
        return api_error(f"Lỗi: {str(e)}", 500)


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
    Train/retrain model for a specific person from data/raw/<name>/ images
    """
    if not database:
        return api_error("Database not initialized", 500)
    
    if not recognizer:
        return api_error("Recognizer not initialized", 500)
    
    try:
        from config import config
        
        # Get images from data/raw/<name>/
        raw_dir = config.DATA_DIR / "raw" / name
        if not raw_dir.exists():
            return api_error(f"No images found for '{name}' in data/raw", 404)
        
        # Read all images
        image_files = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")) + list(raw_dir.glob("*.jpeg"))
        if len(image_files) == 0:
            return api_error(f"No valid images found for '{name}'", 400)
        
        # Process images and extract embeddings
        images = []
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
        
        if len(images) == 0:
            return api_error(f"Failed to load any images for '{name}'", 400)
        
        # Extract embeddings from images using face_analyzer directly
        embeddings = []
        for img in images:
            face_results = face_analyzer.detect_and_align(img)
            for result in face_results:
                embedding = result.get('embedding')
                if embedding is not None:
                    embeddings.append(embedding)
        
        if len(embeddings) == 0:
            return api_error(f"No faces detected in images for '{name}'", 400)
        
        # Save to database
        database.add_person(name, embeddings)
        
        # Update training status
        person_data = database.get_person(name)
        if person_data:
            person_data['metadata']['trained'] = True
            person_data['metadata']['trained_at'] = datetime.now().isoformat()
            person_data['metadata']['image_count'] = len(image_files)
            database.save()
        
        # Reload recognizer database to get latest changes
        recognizer.reload_database()
        
        logger.info(f"Trained {name} with {len(embeddings)} embeddings from {len(image_files)} images")
        
        return api_success(
            data={
                'name': name,
                'images_processed': len(image_files),
                'faces_detected': len(embeddings),
                'embeddings_count': len(embeddings),
                'trained_at': person_data['metadata']['trained_at'] if person_data else None
            },
            message=f"Đã train thành công cho {name} với {len(embeddings)} embeddings từ {len(image_files)} ảnh"
        )
    except Exception as e:
        logger.error(f"Failed to train {name}: {e}", exc_info=True)
        return api_error(f"Train thất bại: {str(e)}", 500)


@api_bp.route('/train-all', methods=['POST'])
def train_all_people():
    """
    Train model for all people in data/raw directory
    """
    if not database:
        return api_error("Database not initialized", 500)
    
    if not recognizer:
        return api_error("Recognizer not initialized", 500)
    
    try:
        from config import config
        raw_dir = config.DATA_DIR / "raw"
        
        if not raw_dir.exists():
            return api_error("data/raw directory not found", 404)
        
        results = []
        errors = []
        total_images = 0
        total_faces = 0
        
        # Process each person directory
        for person_dir in sorted(raw_dir.iterdir()):
            if not person_dir.is_dir() or person_dir.name.startswith('.'):
                continue
            
            name = person_dir.name
            try:
                # Get images
                image_files = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")) + list(person_dir.glob("*.jpeg"))
                if len(image_files) == 0:
                    errors.append({'name': name, 'error': 'No images'})
                    continue
                
                total_images += len(image_files)
                
                # Read and process images
                images = []
                for img_path in image_files:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        images.append(img)
                
                if len(images) == 0:
                    errors.append({'name': name, 'error': 'Failed to load images'})
                    continue
                
                # Extract embeddings
                embeddings = []
                for img in images:
                    face_results = face_analyzer.detect_and_align(img)
                    for result in face_results:
                        embedding = result.get('embedding')
                        if embedding is not None:
                            embeddings.append(embedding)
                
                if len(embeddings) == 0:
                    errors.append({'name': name, 'error': 'No faces detected'})
                    continue
                
                total_faces += len(embeddings)
                
                # Save to database
                database.add_person(name, embeddings)
                
                # Update training status
                person_data = database.get_person(name)
                if person_data:
                    person_data['metadata']['trained'] = True
                    person_data['metadata']['trained_at'] = datetime.now().isoformat()
                    person_data['metadata']['image_count'] = len(image_files)
                
                results.append({
                    'name': name,
                    'images_processed': len(image_files),
                    'faces_detected': len(embeddings),
                    'status': 'success'
                })
            except Exception as e:
                errors.append({'name': name, 'error': str(e)})
                logger.error(f"Failed to train {name}: {e}")
        
        database.save()
        
        # Reload recognizer database to get latest changes
        recognizer.reload_database()
        
        return api_success(
            data={
                'total': len(results) + len(errors),
                'success': len(results),
                'failed': len(errors),
                'total_images': total_images,
                'total_faces': total_faces,
                'results': results,
                'errors': errors
            },
            message=f"Đã train {len(results)}/{len(results) + len(errors)} người với {total_faces} embeddings từ {total_images} ảnh"
        )
    except Exception as e:
        logger.error(f"Failed to train all: {e}", exc_info=True)
        return api_error(f"Lỗi: {str(e)}", 500)


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
