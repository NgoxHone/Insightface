"""
Data preparation pipeline for face recognition training
"""
import logging
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A

from models.face_analyzer import face_analyzer
from config import config

logger = logging.getLogger(__name__)


class FaceDataCollector:
    """Collect and preprocess face images for training"""

    def __init__(self, source_dir: Path, output_dir: Path):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.aligned_dir = self.output_dir / "aligned"
        self.aligned_dir.mkdir(exist_ok=True)

        self.stats = {
            'total_images': 0,
            'faces_detected': 0,
            'faces_saved': 0,
            'skipped_no_face': 0,
            'skipped_low_quality': 0,
            'skipped_error': 0
        }

    def collect_from_folder_structure(
        self,
        min_faces_per_person: int = 5
    ) -> Dict[str, List[Path]]:
        """
        Collect face images organized by person folders

        Expected structure:
            source_dir/
                Person1/
                    img1.jpg
                    img2.jpg
                Person2/
                    ...

        Returns:
            Dict mapping person names to list of face image paths
        """
        logger.info(f"Collecting from {self.source_dir}")

        person_faces: Dict[str, List[Path]] = {}
        person_names = [d for d in self.source_dir.iterdir() if d.is_dir()]

        for person_dir in tqdm(person_names, desc="Processing people"):
            person_name = person_dir.name
            image_paths = []

            # Get all images
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_paths.extend(person_dir.glob(ext))

            if not image_paths:
                logger.warning(f"No images found for {person_name}")
                continue

            self.stats['total_images'] += len(image_paths)

            # Process images
            aligned_paths = self._process_person_images(person_name, image_paths)

            if len(aligned_paths) >= min_faces_per_person:
                person_faces[person_name] = aligned_paths
                logger.info(f"{person_name}: {len(aligned_paths)} faces ready")
            else:
                logger.warning(f"{person_name}: Only {len(aligned_paths)} faces (need {min_faces_per_person})")

        logger.info(f"Collection complete: {self.stats}")
        return person_faces

    def _process_person_images(
        self,
        person_name: str,
        image_paths: List[Path]
    ) -> List[Path]:
        """
        Process images for one person: detect, align, save

        Returns:
            List of aligned face file paths
        """
        person_aligned_dir = self.aligned_dir / person_name
        person_aligned_dir.mkdir(exist_ok=True)

        saved_paths = []

        for img_path in image_paths:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    self.stats['skipped_error'] += 1
                    continue

                # Detect faces
                faces = face_analyzer.detect_faces(img)
                if not faces:
                    self.stats['skipped_no_face'] += 1
                    continue

                self.stats['faces_detected'] += 1

                # Select largest face
                largest_face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                )

                # Align
                landmarks = largest_face.kps
                if landmarks is not None:
                    aligned_face = face_analyzer.align_face(img, landmarks)
                else:
                    # Fallback: crop bbox and resize
                    bbox = largest_face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    aligned_face = cv2.resize(img[y1:y2, x1:x2], (112, 112))

                # Check quality (optional)
                # from recognition.quality_check import assess_face_quality
                # quality = assess_face_quality(aligned_face, landmarks)
                # if quality < config.MIN_IMAGE_QUALITY_SCORE:
                #     self.stats['skipped_low_quality'] += 1
                #     continue

                # Save
                save_path = person_aligned_dir / f"{img_path.stem}_aligned.jpg"
                cv2.imwrite(str(save_path), aligned_face)
                saved_paths.append(save_path)
                self.stats['faces_saved'] += 1

            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                self.stats['skipped_error'] += 1

        return saved_paths


def augment_face(
    face_image: np.ndarray,
    augmentations: Optional[Dict] = None
) -> List[np.ndarray]:
    """
    Apply augmentations to a face image

    Args:
        face_image: Aligned face (112x112)
        augmentations: Augmentation config dict

    Returns:
        List of augmented images (including original if return_original=True)
    """
    if augmentations is None:
        augmentations = config.AUGMENTATION_CONFIG

    augmentations_list = []
    if augmentations.get('flip_horizontal', False):
        augmentations_list.append(A.HorizontalFlip(p=0.5))

    if augmentations.get('rotate_degrees', 0) > 0:
        augmentations_list.append(A.Rotate(limit=augmentations['rotate_degrees'], p=0.5))

    if augmentations.get('color_jitter', 0) > 0:
        augmentations_list.append(A.ColorJitter(
            brightness=augmentations['color_jitter'],
            contrast=augmentations['color_jitter'],
            saturation=augmentations['color_jitter'],
            hue=0.1,
            p=0.5
        ))

    if augmentations.get('blur_prob', 0) > 0:
        augmentations_list.append(A.Blur(blur_limit=3, p=augmentations['blur_prob']))

    if augmentations.get('noise_prob', 0) > 0:
        augmentations_list.append(A.GaussNoise(p=augmentations['noise_prob']))

    if augmentations.get('sharpness_range'):
        min_v, max_v = augmentations['sharpness_range']
        augmentations_list.append(A.RandomBrightnessContrast(p=0.3))

    # Build transform
    transform = A.Compose(augmentations_list) if augmentations_list else None

    # Generate augmented versions
    augmented = [face_image]  # Include original

    if transform:
        # Generate 3-5 augmented versions per image
        n_aug = min(5, len(augmentations_list) + 2)
        for _ in range(n_aug):
            try:
                aug_img = transform(image=face_image)['image']
                augmented.append(aug_img)
            except Exception as e:
                logger.warning(f"Augmentation failed: {e}")
                continue

    return augmented[:6]  # Max 6 versions


def create_dataset_manifest(
    person_faces: Dict[str, List[Path]],
    output_path: Path,
    apply_augmentation: bool = True
) -> pd.DataFrame:
    """
    Create dataset manifest (CSV) with image paths and labels

    Args:
        person_faces: Dict from collect_from_folder_structure
        output_path: Path to save manifest CSV
        apply_augmentation: Whether to include augmented versions

    Returns:
        Pandas DataFrame with 'image_path' and 'label' columns
    """
    import pandas as pd

    data = []

    for label, (person_name, face_paths) in enumerate(person_faces.items()):
        for face_path in face_paths:
            # Add original
            data.append({
                'image_path': str(face_path),
                'label': label,
                'person_name': person_name
            })

            # Generate augmented versions if enabled
            if apply_augmentation:
                try:
                    img = cv2.imread(str(face_path))
                    if img is not None:
                        augmented = augment_face(img)
                        for i, aug_img in enumerate(augmented[1:], 1):  # Skip original
                            aug_path = face_path.parent / f"{face_path.stem}_aug{i}.jpg"
                            cv2.imwrite(str(aug_path), aug_img)
                            data.append({
                                'image_path': str(aug_path),
                                'label': label,
                                'person_name': person_name
                            })
                except Exception as e:
                    logger.warning(f"Failed to augment {face_path}: {e}")

    df = pd.DataFrame(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Created manifest with {len(df)} samples at {output_path}")
    return df


def split_dataset_manifest(
    manifest_path: Path,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[Path, Path]:
    """
    Split dataset manifest into train and val

    Args:
        manifest_path: Path to full manifest CSV
        train_ratio: Fraction for training set
        seed: Random seed

    Returns:
        (train_path, val_path)
    """
    import pandas as pd

    df = pd.read_csv(manifest_path)

    # Stratified split by label
    np.random.seed(seed)
    train_dfs = []
    val_dfs = []

    for label in sorted(df['label'].unique()):
        label_df = df[df['label'] == label]
        n_samples = len(label_df)
        n_train = int(n_samples * train_ratio)

        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_dfs.append(label_df.iloc[train_indices])
        val_dfs.append(label_df.iloc[val_indices])

    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)

    train_path = manifest_path.parent / "train_manifest.csv"
    val_path = manifest_path.parent / "val_manifest.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info(f"Split: {len(train_df)} train, {len(val_df)} val")
    return train_path, val_path


def prepare_training_data(
    source_dir: Path,
    output_dir: Path,
    min_faces_per_person: int = 5,
    apply_augmentation: bool = True
) -> Tuple[Path, Path]:
    """
    Complete data preparation pipeline

    Args:
        source_dir: Raw images organized by person
        output_dir: Where to save processed data
        min_faces_per_person: Minimum face images required
        apply_augmentation: Whether to augment

    Returns:
        (train_manifest_path, val_manifest_path)
    """
    logger.info("Starting data preparation...")

    # 1. Collect and align faces
    collector = FaceDataCollector(source_dir, output_dir)
    person_faces = collector.collect_from_folder_structure(min_faces_per_person)

    if len(person_faces) < 2:
        raise ValueError(f"Need at least 2 people with {min_faces_per_person} faces each. Got {len(person_faces)}")

    # 2. Create manifest
    manifest_path = output_dir / "manifest.csv"
    create_dataset_manifest(person_faces, manifest_path, apply_augmentation)

    # 3. Split train/val
    train_path, val_path = split_dataset_manifest(manifest_path)

    logger.info(f"Data preparation complete!")
    logger.info(f"  Train: {train_path}")
    logger.info(f"  Val: {val_path}")

    return train_path, val_path
