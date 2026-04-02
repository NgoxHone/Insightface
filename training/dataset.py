"""
PyTorch Dataset for Face Recognition Training
"""
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
from albumentations import Compose
import albumentations as A

from ..recognition.utils import normalize_image, load_image_safe
from config import config

logger = logging.getLogger(__name__)


class FaceDataset(Dataset):
    """
    Dataset for training ArcFace model

    Returns:
        - image: normalized tensor [C, H, W]
        - label: integer class label
    """

    def __init__(
        self,
        manifest_path: Path,
        transform: Optional[Compose] = None,
        img_size: Tuple[int, int] = (112, 112),
        training: bool = True
    ):
        """
        Args:
            manifest_path: Path to CSV with 'image_path' and 'label' columns
            transform: Albumentations transform
            img_size: Target image size
            training: Whether in training mode (affects augmentations)
        """
        self.manifest_path = Path(manifest_path)
        self.img_size = img_size
        self.training = training

        # Load manifest
        self.df = pd.read_csv(self.manifest_path)
        logger.info(f"Loaded dataset: {len(self.df)} samples from {self.manifest_path}")

        # Build label mapping
        self.label_to_person = dict(enumerate(sorted(self.df['person_name'].unique())))
        self.person_to_label = {v: k for k, v in self.label_to_person.items()}

        # Verify all images exist
        self._verify_images()

        # Default augmentations if none provided
        if transform is None and training:
            transform = self._get_default_augmentations()
        self.transform = transform

        logger.info(f"Dataset initialized: {len(self)} samples, {len(self.label_to_person)} classes")

    def _verify_images(self):
        """Remove entries with missing images"""
        valid_rows = []
        for _, row in self.df.iterrows():
            img_path = Path(row['image_path'])
            if img_path.exists():
                valid_rows.append(row)
            else:
                logger.warning(f"Image not found: {img_path}")

        if len(valid_rows) < len(self.df):
            logger.warning(f"Removed {len(self.df) - len(valid_rows)} missing images")
            self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

    def _get_default_augmentations(self) -> Compose:
        """Get default training augmentations"""
        aug_config = config.AUGMENTATION_CONFIG

        transforms = [
            A.Resize(height=self.img_size[1], width=self.img_size[0]),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ]

        # Add training augmentations
        if aug_config.get('flip_horizontal', False):
            transforms.insert(0, A.HorizontalFlip(p=0.5))

        if aug_config.get('rotate_degrees', 0) > 0:
            transforms.insert(0, A.Rotate(limit=aug_config['rotate_degrees'], p=0.5))

        if aug_config.get('color_jitter', 0) > 0:
            cj = aug_config['color_jitter']
            transforms.insert(0, A.ColorJitter(
                brightness=cj, contrast=cj, saturation=cj, hue=0.1, p=0.5
            ))

        if aug_config.get('blur_prob', 0) > 0:
            transforms.insert(0, A.Blur(blur_limit=3, p=aug_config['blur_prob']))

        if aug_config.get('noise_prob', 0) > 0:
            transforms.insert(0, A.GaussNoise(p=aug_config['noise_prob']))

        return Compose(transforms)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        """
        Get sample

        Returns:
            image: torch.Tensor [C, H, W]
            label: int
        """
        row = self.df.iloc[idx]
        img_path = Path(row['image_path'])
        label = int(row['label'])

        # Load image
        img = load_image_safe(str(img_path))
        if img is None:
            # Return blank image if load fails
            img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

        # Ensure RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        else:
            # Just resize and normalize
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
            img = normalize_image(img, target_size=self.img_size, normalize=True)

        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC to CHW

        return img_tensor, label

    def get_num_classes(self) -> int:
        return len(self.label_to_person)

    def get_class_names(self) -> list:
        return [self.label_to_person[i] for i in range(len(self.label_to_person))]

    def get_label_counts(self) -> Dict[int, int]:
        """Get sample count per class"""
        return self.df['label'].value_counts().to_dict()

    def balance_sampler_weights(self) -> np.ndarray:
        """
        Compute weights for BalancedSampler
        More samples from minority classes
        """
        label_counts = self.get_label_counts()
        total_samples = len(self.df)
        num_classes = len(label_counts)

        weights = {}
        for label, count in label_counts.items():
            weight = total_samples / (num_classes * count)
            weights[label] = weight

        sample_weights = np.array([weights[row['label']] for _, row in self.df.iterrows()])
        return sample_weights


class TripletFaceDataset(FaceDataset):
    """
    Dataset that returns (anchor, positive, negative) triplets for metric learning
    Useful for training triplet loss
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Group images by label
        self.label_to_indices = {}
        for idx, row in self.df.iterrows():
            label = int(row['label'])
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        # Ensure each class has at least 2 images
        self.valid_labels = [lbl for lbl, indices in self.label_to_indices.items() if len(indices) >= 2]
        if not self.valid_labels:
            raise ValueError("No classes with at least 2 images for triplet sampling")

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            (anchor_img, positive_img, negative_img)
        """
        # Select anchor class (random valid label)
        anchor_label = np.random.choice(self.valid_labels)
        anchor_indices = self.label_to_indices[anchor_label]

        # Select anchor and positive from same class
        anchor_idx, positive_idx = np.random.choice(anchor_indices, size=2, replace=False)

        # Select negative from different class
        negative_label = np.random.choice([lbl for lbl in self.valid_labels if lbl != anchor_label])
        negative_idx = np.random.choice(self.label_to_indices[negative_label])

        # Load images
        anchor_img, _ = super().__getitem__(anchor_idx)
        positive_img, _ = super().__getitem__(positive_idx)
        negative_img, _ = super().__getitem__(negative_idx)

        return anchor_img, positive_img, negative_img
