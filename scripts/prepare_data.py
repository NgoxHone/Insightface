#!/usr/bin/env python3
"""
Data Preparation Script

Usage:
    python scripts/prepare_data.py --source data/raw --output data/processed
    python scripts/prepare_data.py --source data/raw --output data/processed --no-augment
"""
import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from training.data_preparation import prepare_training_data, FaceDataCollector
from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare face dataset for training'
    )
    parser.add_argument(
        '--source', '-s',
        type=Path,
        required=True,
        help='Source directory with person folders (each folder = one person)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=config.DATA_DIR / 'processed',
        help='Output directory for processed faces'
    )
    parser.add_argument(
        '--min-faces',
        type=int,
        default=5,
        help='Minimum faces per person (default: 5)'
    )
    parser.add_argument(
        '--no-augment',
        action='store_true',
        help='Disable data augmentation'
    )
    parser.add_argument(
        '--collect-only',
        action='store_true',
        help='Only collect and align faces, no augmentation/splitting'
    )

    args = parser.parse_args()

    if not args.source.exists():
        logger.error(f"Source directory does not exist: {args.source}")
        return 1

    try:
        if args.collect_only:
            # Only collect and align
            collector = FaceDataCollector(args.source, args.output)
            person_faces = collector.collect_from_folder_structure(
                min_faces_per_person=args.min_faces
            )
            logger.info(f"Collection complete: {len(person_faces)} people")
            logger.info(f"Stats: {collector.stats}")
        else:
            # Full pipeline
            logger.info("Starting full data preparation pipeline...")
            train_path, val_path = prepare_training_data(
                source_dir=args.source,
                output_dir=args.output,
                min_faces_per_person=args.min_faces,
                apply_augmentation=not args.no_augment
            )
            logger.info(f"✓ Prepared dataset:")
            logger.info(f"  Train manifest: {train_path}")
            logger.info(f"  Val manifest: {val_path}")

            # Show summary
            import pandas as pd
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            logger.info(f"  Train samples: {len(train_df)}")
            logger.info(f"  Val samples: {len(val_df)}")
            logger.info(f"  Classes: {len(train_df['label'].unique())}")

        return 0

    except Exception as e:
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
