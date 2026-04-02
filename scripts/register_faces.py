#!/usr/bin/env python3
"""
Face Registration Script

Usage:
    # Register from image files
    python scripts/register_faces.py --name "John Doe" --images /path/to/john/*.jpg

    # Register from webcam
    python scripts/register_faces.py --name "Alice" --webcam --count 50
"""
import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import time

import cv2
import numpy as np

from recognition.face_recognizer import FaceRecognizer
from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Register a new person with face images'
    )
    parser.add_argument(
        '--name', '-n',
        required=True,
        help='Person name'
    )
    parser.add_argument(
        '--images', '-i',
        nargs='+',
        type=Path,
        help='List of image files'
    )
    parser.add_argument(
        '--image-dir',
        type=Path,
        help='Directory containing images'
    )
    parser.add_argument(
        '--webcam',
        action='store_true',
        help='Capture from webcam'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=30,
        help='Number of images to capture from webcam'
    )
    parser.add_argument(
        '--camera-index',
        type=int,
        default=0,
        help='Webcam device index'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=config.UPLOAD_DIR / 'faces',
        help='Directory to save captured face crops'
    )

    args = parser.parse_args()

    # Initialize recognizer
    logger.info("Initializing face recognizer...")
    recognizer = FaceRecognizer(
        database_path=config.DATABASE_PATH,
        threshold=config.RECOGNITION_THRESHOLD
    )
    logger.info("✓ Ready")

    images = []

    if args.webcam:
        # Capture from webcam
        logger.info(f"Opening webcam (index {args.camera_index})...")
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            logger.error("Could not open webcam")
            return 1

        logger.info(f"Capturing {args.count} images. Press SPACE to capture, ESC to quit early.")
        captured = []
        last_capture = 0
        capture_delay = 0.5  # seconds between auto-captures

        while len(captured) < args.count:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break

            # Detect faces for preview
            faces = recognizer.detect(frame)
            preview = frame.copy()

            # Draw faces
            for face in faces:
                bbox = face['bbox']
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if face['quality_score'] > 0.6 else (0, 165, 255)
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)

            # Display info
            cv2.putText(
                preview,
                f"Captured: {len(captured)}/{args.count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )
            cv2.putText(
                preview,
                "SPACE: capture | ESC: quit",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow('Webcam - Press SPACE to capture', preview)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                logger.info("Capture cancelled by user")
                break
            elif key == 32:  # SPACE
                if faces:
                    # Save faces
                    for face in faces:
                        x1, y1, x2, y2 = face['bbox']
                        face_crop = frame[y1:y2, x1:x2]
                        captured.append(face_crop)
                        logger.info(f"Captured face #{len(captured)} (quality: {face['quality_score']:.2f})")

                        # Save to disk
                        if args.output_dir:
                            args.output_dir.mkdir(parents=True, exist_ok=True)
                            filename = args.output_dir / f"{args.name}_{len(captured):03d}.jpg"
                            cv2.imwrite(str(filename), face_crop)
                else:
                    logger.warning("No face detected in current frame")

        cap.release()
        cv2.destroyAllWindows()

        images = captured

    elif args.images:
        # Load from image files
        logger.info(f"Loading {len(args.images)} image files...")
        for img_path in args.images:
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load: {img_path}")
                continue
            images.append(img)
        logger.info(f"Loaded {len(images)} images")

    elif args.image_dir:
        # Load all images from directory
        if not args.image_dir.exists():
            logger.error(f"Directory not found: {args.image_dir}")
            return 1
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        img_paths = [p for p in args.image_dir.iterdir() if p.suffix.lower() in img_extensions]
        logger.info(f"Found {len(img_paths)} images in {args.image_dir}")

        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
        logger.info(f"Loaded {len(images)} images")

    else:
        logger.error("Must provide either --images, --image-dir, or --webcam")
        parser.print_help()
        return 1

    if not images:
        logger.error("No images available for registration")
        return 1

    # Register
    logger.info(f"Registering '{args.name}' with {len(images)} face images...")
    result = recognizer.register_person(args.name, images)

    if result['success']:
        logger.info("✓ Registration successful!")
        logger.info(f"  Name: {result['name']}")
        logger.info(f"  Embeddings extracted: {result['embeddings_extracted']}")
        logger.info(f"  Average quality: {result.get('average_quality', 'N/A'):.2%}")
        logger.info(f"  Total embeddings for person: {result.get('total_embeddings_for_person', 'N/A')}")
        return 0
    else:
        logger.error(f"✗ Registration failed: {result['error']}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
