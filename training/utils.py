"""
Training utilities: logging, metrics, visualization, etc.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional
import json
import csv
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from config import config

logger = logging.getLogger(__name__)


class TrainingLogger:
    """Simple training logger to CSV/JSON"""

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = Path(log_dir) if log_dir else config.TRAINED_MODELS_DIR / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"training_{timestamp}.csv"
        self.json_path = self.log_dir / f"training_{timestamp}.json"

        self.fields = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'timestamp']
        self.csv_file = None
        self.csv_writer = None

        self._open_csv()

    def _open_csv(self):
        """Open CSV file and write header"""
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fields)
        self.csv_writer.writeheader()

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        lr: Optional[float] = None
    ):
        """Log epoch metrics"""
        entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss if val_loss is not None else '',
            'val_acc': val_acc if val_acc is not None else '',
            'lr': lr or '',
            'timestamp': datetime.now().isoformat()
        }

        self.csv_writer.writerow(entry)
        self.csv_file.flush()

    def save_json(self, history: Dict):
        """Save full history to JSON"""
        with open(self.json_path, 'w') as f:
            json.dump(history, f, indent=2)

    def close(self):
        """Close logging files"""
        if self.csv_file:
            self.csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    embeddings: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics

    Args:
        predictions: Predicted class indices
        labels: True class indices
        embeddings: Feature vectors (optional, for additional metrics)

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }

    # Compute embedding quality metrics if provided
    if embeddings is not None:
        intra_class_var = compute_intra_class_variance(embeddings, labels)
        inter_class_dist = compute_inter_class_distance(embeddings, labels)
        metrics['intra_class_variance'] = float(intra_class_var)
        metrics['inter_class_distance'] = float(inter_class_dist)
        metrics['separability'] = float(inter_class_dist / (intra_class_var + 1e-8))

    return metrics


def compute_intra_class_variance(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute average intra-class variance"""
    variances = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        class_embs = embeddings[labels == label]
        if len(class_embs) > 1:
            centroid = class_embs.mean(axis=0)
            var = ((class_embs - centroid) ** 2).sum(axis=1).mean()
            variances.append(var)

    return np.mean(variances) if variances else 0.0


def compute_inter_class_distance(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute average inter-class distance (centroid to centroid)"""
    unique_labels = np.unique(labels)
    centroids = []

    for label in unique_labels:
        class_embs = embeddings[labels == label]
        centroids.append(class_embs.mean(axis=0))

    if len(centroids) < 2:
        return 0.0

    distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            distances.append(dist)

    return np.mean(distances)


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Optional[Path] = None,
    method: str = 'tsne',
    max_samples: int = 1000
) -> Path:
    """
    Visualize embeddings in 2D

    Args:
        embeddings: Feature vectors [N, D]
        labels: Class labels [N]
        output_path: Where to save plot
        method: 'tsne' or 'pca'
        max_samples: Maximum samples to plot

    Returns:
        Path to saved plot
    """
    output_path = output_path or config.TRAINED_MODELS_DIR / "logs" / f"embeddings_{method}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Subsample if too many
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]

    # Dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=min(30, len(embeddings) // 10), random_state=42)
    else:
        reducer = PCA(n_components=2, random_state=42)

    reduced = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)

    # Use colormap
    cmap = plt.cm.tab20 if len(unique_labels) <= 20 else plt.cm.tab20c

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            label=f'Class {label}',
            alpha=0.6,
            s=30,
            color=cmap(idx % cmap.N)
        )

    plt.title(f'Embedding Visualization ({method.upper()})')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved embedding visualization to {output_path}")
    return output_path


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: Optional[Path] = None
) -> Path:
    """
    Plot training and validation curves

    Args:
        history: Dict with 'train_loss', 'train_acc', 'val_loss', 'val_acc' lists
        output_path: Where to save plot

    Returns:
        Path to saved plot
    """
    output_path = output_path or config.TRAINED_MODELS_DIR / "logs" / "training_history.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved training history plot to {output_path}")
    return output_path


def save_model_info(
    model_path: Path,
    dataset_info: Dict,
    metrics: Dict,
    config_info: Dict,
    output_path: Optional[Path] = None
):
    """Save model metadata"""
    output_path = output_path or model_path.parent / "model_info.json"

    info = {
        'model_path': str(model_path),
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset_info,
        'metrics': metrics,
        'config': config_info
    }

    with open(output_path, 'w') as f:
        json.dump(info, f, indent=2)

    logger.info(f"Saved model info to {output_path}")
