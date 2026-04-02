#!/usr/bin/env python3
"""
Benchmark script for face recognition system

Usage:
    python scripts/benchmark.py --test-data data/val/ --model trained_models/model.onnx
    python scripts/benchmark.py --mode inference
"""
import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
from tqdm import tqdm

from models.face_analyzer import face_analyzer
from recognition.face_recognizer import FaceRecognizer
from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def benchmark_inference_speed(
    test_images: List[np.ndarray],
    warmup_iterations: int = 5,
    iterations: int = 50
) -> dict:
    """
    Benchmark inference speed

    Args:
        test_images: List of test images
        warmup_iterations: Warmup runs
        iterations: Benchmark iterations

    Returns:
        Dictionary of metrics
    """
    logger.info(f"Benchmarking inference speed ({iterations} iterations, {len(test_images)} test images)...")

    # Warmup
    for _ in range(warmup_iterations):
        for img in test_images[:3]:  # Use first 3 for warmup
            face_analyzer.detect_faces(img)

    # Benchmark detection
    detect_times = []
    for _ in range(iterations):
        img = test_images[np.random.randint(len(test_images))]
        start = time.perf_counter()
        faces = face_analyzer.detect_faces(img)
        elapsed = time.perf_counter() - start
        detect_times.append(elapsed)

    # Benchmark full recognition (detect + recognize)
    recog = FaceRecognizer()
    recog_times = []
    for _ in range(iterations):
        img = test_images[np.random.randint(len(test_images))]
        start = time.perf_counter()
        result = recog.recognize(img)
        elapsed = time.perf_counter() - start
        recog_times.append(elapsed)

    detect_mean = np.mean(detect_times) * 1000
    detect_std = np.std(detect_times) * 1000
    recog_mean = np.mean(recog_times) * 1000
    recog_std = np.std(recog_times) * 1000

    return {
        'detect': {
            'mean_ms': detect_mean,
            'std_ms': detect_std,
            'fps': 1000 / detect_mean
        },
        'recognize': {
            'mean_ms': recog_mean,
            'std_ms': recog_std,
            'fps': 1000 / recog_mean
        }
    }


def benchmark_database_search(
    db_path: Path,
    query_embeddings: int = 100,
    db_size: int = None
) -> dict:
    """
    Benchmark database search speed

    Args:
        db_path: Path to database
        query_embeddings: Number of query embeddings to test
        db_size: Override database size (for testing)

    Returns:
        Dictionary of metrics
    """
    from recognition.database import FaceDatabase

    logger.info("Benchmarking database search...")
    db = FaceDatabase(db_path)

    if not db._data:
        logger.warning("Database is empty")
        return {}

    # Get all embeddings
    all_embeddings = []
    for name, record in db._data.items():
        all_embeddings.extend(record['embeddings'])

    if not all_embeddings:
        return {}

    actual_db_size = len(all_embeddings)
    logger.info(f"Database has {actual_db_size} embeddings from {len(db._data)} people")

    # Generate query embeddings (random)
    query_embeddings_list = [all_embeddings[i % len(all_embeddings)] for i in range(query_embeddings)]

    # Warmup
    for _ in range(5):
        db.search(query_embeddings_list[0])

    # Benchmark
    times = []
    for q_emb in tqdm(query_embeddings_list, desc="Searching"):
        start = time.perf_counter()
        results = db.search(q_emb, top_k=5)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    mean_time = np.mean(times) * 1000
    std_time = np.std(times) * 1000

    return {
        'db_size': actual_db_size,
        'people': len(db._data),
        'mean_ms': mean_time,
        'std_ms': std_time,
        'queries_per_second': 1000 / mean_time
    }


def benchmark_memory_usage():
    """Estimate memory usage"""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024
    }


def load_test_images(
    data_dir: Path,
    max_images: int = 100
) -> List[np.ndarray]:
    """
    Load test images from directory

    Args:
        data_dir: Directory with images
        max_images: Maximum images to load

    Returns:
        List of images
    """
    images = []
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    img_files = [f for f in data_dir.iterdir() if f.suffix.lower() in extensions]
    img_files = img_files[:max_images]

    logger.info(f"Loading {len(img_files)} test images from {data_dir}")

    for img_path in tqdm(img_files, desc="Loading images"):
        img = cv2.imread(str(img_path))
        if img is not None and img.size > 0:
            images.append(img)

    logger.info(f"Loaded {len(images)} valid images")
    return images


def print_benchmark_report(results: dict):
    """Print formatted benchmark report"""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    if 'inference' in results:
        print("\nInference Speed:")
        inf = results['inference']
        print(f"  Detection:  {inf['detect']['mean_ms']:.2f} ± {inf['detect']['std_ms']:.2f} ms  ({inf['detect']['fps']:.1f} FPS)")
        print(f"  Recognize:  {inf['recognize']['mean_ms']:.2f} ± {inf['recognize']['std_ms']:.2f} ms  ({inf['recognize']['fps']:.1f} FPS)")

    if 'database' in results:
        print("\nDatabase Search:")
        db = results['database']
        print(f"  Database size: {db['db_size']} embeddings, {db['people']} people")
        print(f"  Search time:   {db['mean_ms']:.3f} ± {db['std_ms']:.3f} ms")
        print(f"  Search QPS:    {db['queries_per_second']:.0f} queries/sec")

    if 'memory' in results:
        print("\nMemory Usage:")
        mem = results['memory']
        print(f"  RSS:          {mem['rss_mb']:.1f} MB")
        print(f"  VMS:          {mem['vms_mb']:.1f} MB")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Benchmark face recognition system')
    parser.add_argument('--test-data', type=Path, help='Directory with test images')
    parser.add_argument('--model', type=Path, help='Model path (optional)')
    parser.add_argument('--db', type=Path, default=config.DATABASE_PATH, help='Database path')
    parser.add_argument('--mode', choices=['inference', 'database', 'all'], default='all', help='Benchmark mode')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations')
    parser.add_argument('--max-images', type=int, default=100, help='Max test images to load')

    args = parser.parse_args()

    results = {}

    try:
        # Benchmark inference
        if args.mode in ['inference', 'all']:
            if not args.test_data:
                logger.error("--test-data required for inference benchmarking")
                return 1

            test_images = load_test_images(args.test_data, args.max_images)
            if not test_images:
                logger.error("No test images loaded")
                return 1

            inf_results = benchmark_inference_speed(test_images, iterations=args.iterations)
            results['inference'] = inf_results

        # Benchmark database
        if args.mode in ['database', 'all']:
            db_results = benchmark_database_search(args.db)
            if db_results:
                results['database'] = db_results

        # Memory usage
        if args.mode == 'inference' or args.mode == 'all':
            results['memory'] = benchmark_memory_usage()

        # Print report
        print_benchmark_report(results)

        return 0

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted")
        return 0
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
