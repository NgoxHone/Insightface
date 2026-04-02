"""
Face Recognition Model Trainer
Fine-tune ArcFace model on custom dataset
"""
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import pandas as pd

from models.model_utils import compute_similarity
from .dataset import FaceDataset, TripletFaceDataset
from config import config

logger = logging.getLogger(__name__)


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    https://arxiv.org/abs/1801.07698
    """
    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 100,
        s: float = 64.0,
        m: float = 0.5,
        easy_margin: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        # Weight matrix: [out_features, in_features]
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, in_features] normalized embeddings
            labels: [B] class labels

        Returns:
            loss value
        """
        # Normalize weight and features
        weight = F.normalize(self.weight, p=2, dim=1)
        features = F.normalize(features, p=2, dim=1)

        # Cosine similarity
        cos_theta = torch.mm(features, weight.T)  # [B, out_features]

        # Clamp to avoid numerical issues
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # Convert to angle
        theta = torch.acos(cos_theta)

        # Add margin
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        if self.easy_margin:
            cos_theta_m = torch.cos(theta + self.m)
        else:
            cos_theta_m = torch.cos(theta + self.m * one_hot)

        # Apply s * (cos_theta_m - one_hot * cos_theta)
        output = cos_theta * (1 - one_hot) + cos_theta_m * one_hot
        output = output * self.s

        # Cross-entropy loss
        log_probs = F.log_softmax(output, dim=1)
        loss = F.nll_loss(log_probs, labels)

        return loss


class InsightFaceTrainer:
    """Trainer for fine-tuning ArcFace model"""

    def __init__(
        self,
        train_dataset: FaceDataset,
        val_dataset: Optional[FaceDataset] = None,
        model_save_dir: Optional[Path] = None,
        num_classes: Optional[int] = None,
        device: Optional[str] = None,
        use_triplet_loss: bool = False
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_save_dir = Path(model_save_dir) if model_save_dir else config.TRAINED_MODELS_DIR / "finetuned"
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

        self.num_classes = num_classes or train_dataset.get_num_classes()
        self.device = device or self._get_device()
        self.use_triplet_loss = use_triplet_loss

        # Check if we need to expand output layer
        self.base_model = self._load_pretrained_model()
        self.model = self._prepare_model()

        logger.info(f"Trainer initialized: {self.num_classes} classes, device={self.device}")

    def _get_device(self) -> str:
        """Get best available device"""
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Silicon MPS")
        else:
            device = "cpu"
            logger.info("Using CPU")
        return device

    def _load_pretrained_model(self):
        """Load pretrained InsightFace model"""
        try:
            from insightface.model_zoo import get_model

            logger.info(f"Loading pretrained model: {config.MODEL_NAME}")
            model = get_model(name=config.MODEL_NAME, root=str(config.MODELS_DIR))
            model.prepare(ctx_id=0)

            # Extract PyTorch model if available
            if hasattr(model, 'model'):
                torch_model = model.model
            elif hasattr(model, 'net'):
                torch_model = model.net
            else:
                # Try to get backbone + head
                torch_model = model

            return torch_model
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            raise

    def _prepare_model(self) -> nn.Module:
        """
        Prepare model for fine-tuning
        - Adapt output layer for new number of classes
        - Freeze backbone optionally
        """
        model = self.base_model

        # Check model architecture
        logger.info(f"Model type: {type(model)}")

        # Try to find the feature extractor and head
        # ArcFace models typically have a backbone + head structure
        if hasattr(model, 'output_layer'):
            # Replace output layer
            in_features = model.output_layer.in_features
            model.output_layer = nn.Linear(in_features, self.num_classes)
            logger.info(f"Replaced output layer: {in_features} -> {self.num_classes}")
        elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            # ResNet-style
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)
            logger.info(f"Replaced FC layer: {in_features} -> {self.num_classes}")
        else:
            # Try to find the last linear layer
            logger.warning("Could not find standard output layer, freezing backbone only")
            # Assume last module is head
            pass

        # Move to device
        model = model.to(self.device)

        return model

    def _get_dataloaders(self) -> Tuple[DataLoader, Optional[DataLoader]]:
        """Create train and val dataloaders"""
        batch_size = config.TRAIN_BATCH_SIZE

        # Optionally use weighted sampler for class imbalance
        sampler = None
        if hasattr(self.train_dataset, 'balance_sampler_weights'):
            weights = self.train_dataset.balance_sampler_weights()
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
            logger.info("Using weighted sampler for class balance")

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=min(os.cpu_count() or 2, 4),
            pin_memory=True if 'cuda' in self.device else False,
            drop_last=True
        )

        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=min(os.cpu_count() or 2, 4),
                pin_memory=True if 'cuda' in self.device else False
            )

        return train_loader, val_loader

    def train(
        self,
        epochs: int = None,
        learning_rate: float = None,
        save_checkpoint_every: int = None,
        early_stopping_patience: int = None
    ) -> Dict[str, Any]:
        """
        Main training loop

        Returns:
            Training history dictionary
        """
        epochs = epochs or config.TRAIN_EPOCHS
        lr = learning_rate or config.TRAIN_LEARNING_RATE
        save_every = save_checkpoint_every or config.TRAIN_SAVE_CHECKPOINT_EVERY
        patience = early_stopping_patience or config.TRAIN_EARLY_STOPPING_PATIENCE

        logger.info(f"Starting training: {epochs} epochs, lr={lr}, device={self.device}")

        train_loader, val_loader = self._get_dataloaders()

        # Loss function
        if self.use_triplet_loss:
            criterion = nn.TripletMarginLoss(margin=0.2)
        else:
            criterion = ArcFaceLoss(
                in_features=512,
                out_features=self.num_classes,
                s=64.0,
                m=0.5
            ).to(self.device)

        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        # optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        # Scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # Train
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer, epoch
            )
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            # Validate
            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                val_loss, val_acc = self._validate_epoch(val_loader, criterion)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # Check for improvement
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch
                    patience_counter = 0
                    self._save_checkpoint(f"best_model_{epoch+1}.pth")
                else:
                    patience_counter += 1

                logger.info(
                    f"  Train: loss={train_loss:.4f}, acc={train_acc:.2%} | "
                    f"Val: loss={val_loss:.4f}, acc={val_acc:.2%} (best={best_val_acc:.2%})"
                )

                # Early stopping
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                    break
            else:
                logger.info(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2%}")

            # Scheduler step
            scheduler.step()

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth")

            # Save final model
        self._save_checkpoint("final_model.pth")

        logger.info(f"Training complete! Best val accuracy: {best_val_acc:.2%} at epoch {best_epoch + 1}")

        return history

    def _train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Train E{epoch+1}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward
            features = self.model(images)  # [B, 512]
            features = F.normalize(features, p=2, dim=1)

            loss = criterion(features, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            total += labels.size(0)

            # Accuracy: compute nearest neighbor in weight space
            if not self.use_triplet_loss:
                with torch.no_grad():
                    weight = F.normalize(criterion.weight, p=2, dim=1)
                    cos_sim = torch.mm(features, weight.T)
                    preds = cos_sim.argmax(dim=1)
                    correct += (preds == labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def _validate_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Val", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                features = self.model(images)
                features = F.normalize(features, p=2, dim=1)

                loss = criterion(features, labels)
                total_loss += loss.item()
                total += labels.size(0)

                # Accuracy
                weight = F.normalize(criterion.weight, p=2, dim=1)
                cos_sim = torch.mm(features, weight.T)
                preds = cos_sim.argmax(dim=1)
                correct += (preds == labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.model_save_dir / filename

        checkpoint = {
            'epoch': filename,
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'config': {
                'img_size': self.train_dataset.img_size,
                'normalize': True
            }
        }

        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Saved checkpoint: {checkpoint_path}")

    def export_onnx(self, output_path: Optional[Path] = None) -> Path:
        """
        Export model to ONNX format for deployment

        Returns:
            Path to ONNX file
        """
        self.model.eval()

        output_path = output_path or self.model_save_dir / "model.onnx"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Dummy input
        dummy_input = torch.randn(1, 3, 112, 112).to(self.device)

        # Export
        logger.info(f"Exporting model to ONNX: {output_path}")
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['feature'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'feature': {0: 'batch_size'}
            }
        )

        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            logger.info("ONNX model verified successfully")
        except Exception as e:
            logger.warning(f"ONNX verification failed: {e}")

        return output_path


def train_model(
    train_manifest: Path,
    val_manifest: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    use_triplet: bool = False,
    resume_from: Optional[Path] = None
) -> InsightFaceTrainer:
    """
    Convenience function to train a model

    Args:
        train_manifest: Path to train CSV
        val_manifest: Path to val CSV
        output_dir: Where to save model
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        use_triplet: Use triplet loss
        resume_from: Checkpoint to resume from

    Returns:
        Trained InsightFaceTrainer
    """
    # Override config temporarily
    if batch_size:
        config.TRAIN_BATCH_SIZE = batch_size

    # Create datasets
    train_dataset = FaceDataset(train_manifest, training=True)

    val_dataset = None
    if val_manifest and Path(val_manifest).exists():
        val_dataset = FaceDataset(val_manifest, training=False)
        logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    else:
        logger.info(f"Train: {len(train_dataset)} samples (no validation)")

    # Create trainer
    trainer = InsightFaceTrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_save_dir=output_dir,
        use_triplet_loss=use_triplet
    )

    # Resume if checkpoint provided
    if resume_from and resume_from.exists():
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])

    # Train
    history = trainer.train(epochs=epochs, learning_rate=learning_rate)

    # Export to ONNX
    onnx_path = trainer.export_onnx()

    logger.info(f"Training complete! Model saved to {onnx_path}")

    return trainer
