#!/usr/bin/env python3
"""
Training Script - Fine-tune ArcFace model

Usage:
    python scripts/train.py --train-manifest data/splits/train_manifest.csv
    python scripts/train.py --train-manifest data/splits/train_manifest.csv --val-manifest data/splits/val_manifest.csv --epochs 50 --batch-size 32
"""
import argparse
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from training.trainer import train_model, InsightFaceTrainer
from training.dataset import FaceDataset
from training.utils import TrainingLogger, plot_training_history, visualize_embeddings
from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune face recognition model'
    )
    parser.add_argument(
        '--train-manifest',
        type=Path,
        required=True,
        help='Path to training manifest CSV'
    )
    parser.add_argument(
        '--val-manifest',
        type=Path,
        help='Path to validation manifest CSV'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=config.TRAINED_MODELS_DIR / 'finetuned',
        help='Output directory for model checkpoints'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=config.TRAIN_EPOCHS,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=config.TRAIN_BATCH_SIZE,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=config.TRAIN_LEARNING_RATE,
        help='Learning rate'
    )
    parser.add_argument(
        '--resume',
        type=Path,
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--export-onnx',
        action='store_true',
        help='Export final model to ONNX'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate embedding visualization after training'
    )

    args = parser.parse_args()

    # Validate paths
    if not args.train_manifest.exists():
        logger.error(f"Training manifest not found: {args.train_manifest}")
        return 1

    if args.val_manifest and not args.val_manifest.exists():
        logger.warning(f"Validation manifest not found: {args.val_manifest}. Training without validation.")
        args.val_manifest = None

    try:
        logger.info("=" * 60)
        logger.info("Face Recognition Model Training")
        logger.info("=" * 60)
        logger.info(f"Train manifest: {args.train_manifest}")
        logger.info(f"Val manifest: {args.val_manifest}")
        logger.info(f"Output dir: {args.output}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info("=" * 60)

        # Train
        trainer = train_model(
            train_manifest=args.train_manifest,
            val_manifest=args.val_manifest,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            resume_from=args.resume
        )

        logger.info("✓ Training completed successfully!")

        # Optionally visualize embeddings
        if args.visualize:
            logger.info("Generating embedding visualization...")
            # Extract embeddings from validation set
            val_dataset = FaceDataset(args.val_manifest, training=False) if args.val_manifest else None
            if val_dataset:
                import torch
                embeddings = []
                labels = []

                with torch.no_grad():
                    for i in range(min(500, len(val_dataset))):
                        img, label = val_dataset[i]
                        img = img.unsqueeze(0).to(trainer.device)
                        feat = trainer.model(img)
                        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
                        embeddings.append(feat.cpu().numpy().flatten())
                        labels.append(label)

                if embeddings:
                    embeddings = np.array(embeddings)
                    labels = np.array(labels)
                    visualize_embeddings(embeddings, labels, output_path=args.output / "embeddings_tsne.png")

        logger.info(f"Model saved to: {args.output / 'model.onnx'}")
        logger.info("All done!")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
