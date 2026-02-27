"""
Data Preparation Script for HAM10000 Dataset.

This script helps prepare the HAM10000 dataset for training by:
1. Validating the dataset structure
2. Creating the labels CSV file from metadata
3. Performing basic data quality checks
4. Generating dataset statistics

The HAM10000 dataset should be downloaded from:
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

After downloading, organize the data as follows:
data/HAM10000/
    images/
        ISIC_0024306.jpg
        ISIC_0024307.jpg
        ...
    HAM10000_metadata.csv  (original metadata file)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Expected class labels
EXPECTED_CLASSES = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

# Default mapping for common ISIC diagnosis_3 labels to HAM10000 classes.
# This can be extended/overridden via --label-mapping-file.
DEFAULT_ISIC_DIAGNOSIS3_MAPPING = {
    "Actinic keratosis": "akiec",
    "Bowen disease": "akiec",
    "Bowen's disease": "akiec",
    "Squamous cell carcinoma in situ": "akiec",
    "Basal cell carcinoma": "bcc",
    "Benign keratosis": "bkl",
    "Pigmented benign keratosis": "bkl",
    "Lichen planus-like keratosis": "bkl",
    "Seborrheic keratosis": "bkl",
    "Solar lentigo": "bkl",
    "Dermatofibroma": "df",
    "Melanoma": "mel",
    "Melanoma, NOS": "mel",
    "Melanoma In Situ": "mel",
    "Melanoma in situ": "mel",
    "Melanoma Invasive": "mel",
    "Nevus": "nv",
    "Blue nevus": "nv",
    "Clark nevus": "nv",
    "Congenital nevus": "nv",
    "Dermal nevus": "nv",
    "Reed or Spitz nevus": "nv",
    "Squamous cell carcinoma, NOS": "akiec",
    "Vascular lesion": "vasc",
    "Benign soft tissue proliferations - Vascular": "vasc",
    "Angioma": "vasc",
    "Hemangioma": "vasc",
}

# Target class distribution requested for balanced training dataset (total: 19,000)
TARGET_DISTRIBUTION = {
    "mel": 7000,
    "nv": 3000,
    "bcc": 3000,
    "akiec": 1500,
    "bkl": 1500,
    "df": 1500,
    "vasc": 1500,
}


def _get_image_path(images_dir: Path, image_id: str) -> Optional[Path]:
    """Resolve an image path by trying common extensions."""
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_label_mapping(
    label_mapping_file: Optional[Path],
    include_default_isic_mapping: bool,
) -> Dict[str, str]:
    """Load and validate label mapping from defaults and/or JSON file."""
    mapping: Dict[str, str] = {}

    if include_default_isic_mapping:
        mapping.update(DEFAULT_ISIC_DIAGNOSIS3_MAPPING)

    if label_mapping_file is not None:
        if not label_mapping_file.exists():
            raise FileNotFoundError(f"Label mapping file not found: {label_mapping_file}")

        if label_mapping_file.suffix.lower() != ".json":
            raise ValueError(
                "Label mapping file must be JSON (.json) with "
                "{\"raw_label\": \"target_label\"} entries"
            )

        with open(label_mapping_file, "r", encoding="utf-8") as f:
            user_mapping = json.load(f)

        if not isinstance(user_mapping, dict):
            raise ValueError("Label mapping JSON must be an object/dictionary")

        for key, value in user_mapping.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Label mapping keys/values must be strings")
            mapping[key.strip()] = value.strip()

    invalid_targets = sorted({v for v in mapping.values() if v not in EXPECTED_CLASSES})
    if invalid_targets:
        raise ValueError(
            "Label mapping contains targets outside expected classes: "
            f"{invalid_targets}. Expected one of: {sorted(EXPECTED_CLASSES.keys())}"
        )

    return mapping


def _apply_balancing_augmentation(
    image_bgr: np.ndarray,
    rng: np.random.Generator,
    max_rotation_deg: float = 25.0,
    max_shift_fraction: float = 0.15,
    max_shear_fraction: float = 0.15,
    zoom_range: float = 0.4,
    enable_horizontal_flip: bool = True,
    enable_vertical_flip: bool = True,
    brightness_min: float = 0.9,
    brightness_max: float = 1.5,
    clahe_clip_limit: float = 4.0,
) -> np.ndarray:
    """
    Apply geometric matrix transforms + pixel-level adjustments.

    Geometric operations:
    - Rotation up to ±25°
    - Width/height shift up to ±15%
    - Shear up to ±15%
    - Zoom range 0.4 => scale in [0.6, 1.4]
    - Horizontal/vertical flipping enabled
    - Fill mode 'nearest' (edge replication)

    Pixel-level operations:
    - Brightness factor in [0.9, 1.5]
    - CLAHE with clip limit 4.0
    """
    import cv2

    height, width = image_bgr.shape[:2]

    if enable_horizontal_flip and rng.random() < 0.5:
        image_bgr = cv2.flip(image_bgr, 1)
    if enable_vertical_flip and rng.random() < 0.5:
        image_bgr = cv2.flip(image_bgr, 0)

    angle_deg = rng.uniform(-max_rotation_deg, max_rotation_deg)
    angle_rad = math.radians(angle_deg)
    tx = rng.uniform(-max_shift_fraction, max_shift_fraction) * width
    ty = rng.uniform(-max_shift_fraction, max_shift_fraction) * height
    shear = rng.uniform(-max_shear_fraction, max_shear_fraction)
    scale = rng.uniform(max(0.1, 1.0 - zoom_range), 1.0 + zoom_range)

    cx = width / 2.0
    cy = height / 2.0

    c_to_origin = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    c_back = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
    rotation = np.array(
        [
            [math.cos(angle_rad), -math.sin(angle_rad), 0.0],
            [math.sin(angle_rad), math.cos(angle_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    shearing = np.array([[1.0, shear, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    scaling = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]])
    translation = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])

    affine_3x3 = translation @ c_back @ rotation @ shearing @ scaling @ c_to_origin
    affine_2x3 = affine_3x3[:2, :]

    transformed = cv2.warpAffine(
        image_bgr,
        affine_2x3.astype(np.float32),
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REPLICATE,
    )

    brightness_factor = rng.uniform(brightness_min, brightness_max)
    transformed = np.clip(transformed.astype(np.float32) * brightness_factor, 0, 255)
    transformed = transformed.astype(np.uint8)

    lab = cv2.cvtColor(transformed, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    transformed = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return transformed


def build_balanced_augmented_dataset(
    df: pd.DataFrame,
    data_dir: Path,
    output_dir: Path,
    output_csv: Path,
    target_distribution: Dict[str, int],
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a balanced dataset with augmented samples saved to disk.

    The output contains exactly the requested per-class counts and an
    aggregate of 19,000 images based on TARGET_DISTRIBUTION.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for balanced augmentation. Install with: "
            "pip install opencv-python-headless"
        ) from exc

    images_dir = data_dir / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    missing_targets = set(target_distribution.keys()) - set(df["label"].unique())
    if missing_targets:
        raise ValueError(
            f"Target classes missing in metadata: {sorted(missing_targets)}"
        )

    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building balanced augmented dataset")
    logger.info(f"Output directory: {output_dir}")

    rng = np.random.default_rng(seed)
    records = []

    for label, target_count in target_distribution.items():
        class_df = df[df["label"] == label].copy().reset_index(drop=True)
        source_count = len(class_df)

        logger.info(
            f"Class '{label}': source={source_count}, target={target_count}"
        )

        if source_count == 0:
            raise ValueError(f"No samples found for class '{label}'")

        if source_count >= target_count:
            selected_indices = rng.choice(source_count, size=target_count, replace=False)
            selected_df = class_df.iloc[selected_indices].reset_index(drop=True)

            for _, row in selected_df.iterrows():
                src_id = row["image_id"]
                src_path = _get_image_path(images_dir, src_id)
                if src_path is None:
                    continue
                dst_id = src_id
                dst_path = output_images_dir / f"{dst_id}.jpg"
                shutil.copy2(src_path, dst_path)

                out_row = row.copy()
                out_row["image_id"] = dst_id
                records.append(out_row)

            continue

        for _, row in class_df.iterrows():
            src_id = row["image_id"]
            src_path = _get_image_path(images_dir, src_id)
            if src_path is None:
                continue
            dst_id = src_id
            dst_path = output_images_dir / f"{dst_id}.jpg"
            shutil.copy2(src_path, dst_path)

            out_row = row.copy()
            out_row["image_id"] = dst_id
            records.append(out_row)

        needed_augmented = target_count - source_count
        sampled_indices = rng.choice(source_count, size=needed_augmented, replace=True)

        for aug_idx, src_idx in enumerate(
            tqdm(sampled_indices, desc=f"Augmenting {label}", leave=False)
        ):
            row = class_df.iloc[int(src_idx)]
            src_id = row["image_id"]
            src_path = _get_image_path(images_dir, src_id)
            if src_path is None:
                continue

            src_bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
            if src_bgr is None:
                continue

            aug_bgr = _apply_balancing_augmentation(src_bgr, rng=rng)
            dst_id = f"aug_{label}_{aug_idx:05d}_{src_id}"
            dst_path = output_images_dir / f"{dst_id}.jpg"
            cv2.imwrite(str(dst_path), aug_bgr)

            out_row = row.copy()
            out_row["image_id"] = dst_id
            records.append(out_row)

    balanced_df = pd.DataFrame(records)

    class_counts = balanced_df["label"].value_counts().to_dict()
    if class_counts != target_distribution:
        raise RuntimeError(
            "Balanced dataset counts do not match target distribution. "
            f"Actual: {class_counts}, Target: {target_distribution}"
        )

    balanced_df = balanced_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    balanced_df.to_csv(output_csv, index=False)

    logger.info(
        f"Balanced dataset created with {len(balanced_df)} images at: {output_images_dir}"
    )
    logger.info(f"Balanced labels saved to: {output_csv}")

    return balanced_df


def validate_image(image_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that an image file is readable and properly formatted.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Re-open to check if we can load it
        with Image.open(image_path) as img:
            img.load()
            if img.mode not in ["RGB", "RGBA", "L"]:
                return False, f"Unexpected image mode: {img.mode}"
        return True, None
    except Exception as e:
        return False, str(e)


def prepare_dataset(
    data_dir: Path,
    output_csv: Path,
    metadata_file: Optional[Path] = None,
    validate_images: bool = True,
    label_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Prepare the HAM10000 dataset for training.

    Args:
        data_dir: Root data directory (should contain 'images' folder)
        output_csv: Path to save the prepared labels CSV
        metadata_file: Path to original HAM10000_metadata.csv
        validate_images: Whether to validate all images

    Returns:
        Prepared DataFrame
    """
    images_dir = data_dir / "images"

    # Check directory structure
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = [f for f in images_dir.iterdir() if f.suffix in image_extensions]

    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    logger.info(f"Found {len(image_files)} images")

    # Try to load metadata
    if metadata_file is None:
        # Look for common metadata file names
        possible_names = [
            "HAM10000_metadata.csv",
            "metadata.csv",
            "HAM10000_metadata.tab",
        ]
        for name in possible_names:
            candidate = data_dir / name
            if candidate.exists():
                metadata_file = candidate
                break

    if metadata_file is None or not metadata_file.exists():
        logger.warning(
            "Metadata file not found. Creating labels from directory structure or "
            "please ensure images are organized by class or provide metadata file."
        )
        # Create placeholder DataFrame
        df = pd.DataFrame(
            {
                "image_id": [f.stem for f in image_files],
                "label": ["unknown"] * len(image_files),
            }
        )
        logger.warning("Labels set to 'unknown' - please update labels.csv manually")
    else:
        # Load metadata
        logger.info(f"Loading metadata from: {metadata_file}")

        if metadata_file.suffix == ".tab":
            metadata = pd.read_csv(metadata_file, sep="\t")
        else:
            metadata = pd.read_csv(metadata_file)

        # Map columns to expected format
        column_mapping = {
            "image_id": "image_id",
            "isic_id": "image_id",
            "image": "image_id",
            "dx": "label",
            "diagnosis": "label",
            "lesion_id": "lesion_id",
        }

        # Rename columns
        for old_name, new_name in column_mapping.items():
            if old_name in metadata.columns and old_name != new_name:
                metadata = metadata.rename(columns={old_name: new_name})

        if "label" not in metadata.columns:
            if "diagnosis_3" in metadata.columns:
                metadata["label"] = metadata["diagnosis_3"]
            elif "diagnosis_2" in metadata.columns:
                metadata["label"] = metadata["diagnosis_2"]
            elif "diagnosis_1" in metadata.columns:
                metadata["label"] = metadata["diagnosis_1"]

        if "label" in metadata.columns:
            if "diagnosis_2" in metadata.columns:
                metadata["label"] = metadata["label"].fillna(metadata["diagnosis_2"])
            if "diagnosis_1" in metadata.columns:
                metadata["label"] = metadata["label"].fillna(metadata["diagnosis_1"])

        # Validate required columns
        if "image_id" not in metadata.columns or "label" not in metadata.columns:
            raise ValueError(
                f"Metadata must contain 'image_id' and 'label' columns. "
                f"Found: {list(metadata.columns)}"
            )

        # Keep only needed columns
        columns_to_keep = ["image_id", "label"]
        if "lesion_id" in metadata.columns:
            columns_to_keep.append("lesion_id")

        df = metadata[columns_to_keep].copy()

        if label_mapping:
            mapping_normalized = {k.strip(): v.strip() for k, v in label_mapping.items()}
            mapping_casefold = {k.casefold(): v for k, v in mapping_normalized.items()}

            def _map_label(value: object) -> str:
                if pd.isna(value):
                    return "unknown"
                raw = str(value).strip()
                if raw in mapping_normalized:
                    return mapping_normalized[raw]
                return mapping_casefold.get(raw.casefold(), raw)

            before_unique = df["label"].nunique(dropna=False)
            df["label"] = df["label"].apply(_map_label)
            after_unique = df["label"].nunique(dropna=False)
            logger.info(
                "Applied label mapping: %d unique labels -> %d unique labels",
                before_unique,
                after_unique,
            )

        logger.info(f"Loaded metadata for {len(df)} images")

    # Validate labels
    unique_labels = df["label"].unique()
    unknown_labels = set(unique_labels) - set(EXPECTED_CLASSES.keys()) - {"unknown"}
    if unknown_labels:
        logger.warning(f"Unexpected labels found: {unknown_labels}")

    # Match images with metadata
    image_ids_on_disk = {f.stem for f in image_files}
    image_ids_in_metadata = set(df["image_id"])

    # Find mismatches
    missing_in_metadata = image_ids_on_disk - image_ids_in_metadata
    missing_on_disk = image_ids_in_metadata - image_ids_on_disk

    if missing_in_metadata:
        logger.warning(f"{len(missing_in_metadata)} images on disk not in metadata")

    if missing_on_disk:
        logger.warning(f"{len(missing_on_disk)} images in metadata not found on disk")
        # Remove missing images from DataFrame
        df = df[df["image_id"].isin(image_ids_on_disk)]

    # Validate images if requested
    if validate_images:
        logger.info("Validating images...")
        valid_images = []
        invalid_images = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
            image_id = row["image_id"]
            # Find the image file
            image_path = None
            for ext in image_extensions:
                candidate = images_dir / f"{image_id}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                invalid_images.append((image_id, "File not found"))
                continue

            is_valid, error = validate_image(image_path)
            if is_valid:
                valid_images.append(image_id)
            else:
                invalid_images.append((image_id, error))

        if invalid_images:
            logger.warning(f"Found {len(invalid_images)} invalid images")
            for img_id, error in invalid_images[:10]:
                logger.warning(f"  {img_id}: {error}")
            if len(invalid_images) > 10:
                logger.warning(f"  ... and {len(invalid_images) - 10} more")

        # Keep only valid images
        df = df[df["image_id"].isin(valid_images)]

    # Save labels CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved labels to: {output_csv}")

    return df


def print_statistics(df: pd.DataFrame) -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(f"\nTotal samples: {len(df)}")

    print("\nClass distribution:")
    print("-" * 40)

    class_counts = df["label"].value_counts()
    total = len(df)

    for label in sorted(class_counts.index):
        count = class_counts[label]
        pct = count / total * 100
        desc = EXPECTED_CLASSES.get(label, "Unknown")
        print(f"  {label:6s}: {count:5d} ({pct:5.1f}%) - {desc}")

    print("-" * 40)

    # Class imbalance ratio
    max_class = class_counts.max()
    min_class = class_counts.min()
    imbalance_ratio = max_class / min_class
    print(f"\nClass imbalance ratio: {imbalance_ratio:.1f}:1")

    if "lesion_id" in df.columns:
        unique_lesions = df["lesion_id"].nunique()
        images_per_lesion = len(df) / unique_lesions
        print(f"Unique lesions: {unique_lesions}")
        print(f"Average images per lesion: {images_per_lesion:.1f}")

    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare HAM10000 dataset for training"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/HAM10000"),
        help="Root data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output labels CSV path",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to original metadata file",
    )
    parser.add_argument(
        "--label-mapping-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file mapping raw metadata labels to HAM classes "
            "(e.g., diagnosis_3 values to akiec/bcc/bkl/df/mel/nv/vasc)"
        ),
    )
    parser.add_argument(
        "--use-default-isic-mapping",
        action="store_true",
        help=(
            "Apply built-in mapping from common ISIC diagnosis_3 labels to "
            "HAM10000 classes"
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip image validation",
    )
    parser.add_argument(
        "--build-balanced-dataset",
        action="store_true",
        help=(
            "Create balanced augmented dataset (total 19,000) with target "
            "distribution: mel=7000, nv=3000, bcc=3000, akiec=1500, bkl=1500, "
            "df=1500, vasc=1500"
        ),
    )
    parser.add_argument(
        "--balanced-output-dir",
        type=Path,
        default=None,
        help="Output directory for balanced dataset (contains images/ and labels CSV)",
    )
    parser.add_argument(
        "--balanced-output-csv",
        type=Path,
        default=None,
        help="Output labels CSV path for balanced dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible balancing/augmentation",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    def _resolve_project_path(path_value: Optional[Path]) -> Optional[Path]:
        if path_value is None:
            return None
        if path_value.is_absolute():
            return path_value
        return (project_root / path_value).resolve()

    args.data_dir = _resolve_project_path(args.data_dir)
    args.output = _resolve_project_path(args.output)
    args.metadata = _resolve_project_path(args.metadata)
    args.label_mapping_file = _resolve_project_path(args.label_mapping_file)
    args.balanced_output_dir = _resolve_project_path(args.balanced_output_dir)
    args.balanced_output_csv = _resolve_project_path(args.balanced_output_csv)

    if args.output is None:
        args.output = args.data_dir / "labels.csv"

    label_mapping = load_label_mapping(
        label_mapping_file=args.label_mapping_file,
        include_default_isic_mapping=args.use_default_isic_mapping,
    )

    df = prepare_dataset(
        data_dir=args.data_dir,
        output_csv=args.output,
        metadata_file=args.metadata,
        validate_images=not args.skip_validation,
        label_mapping=label_mapping,
    )

    print_statistics(df)

    if args.build_balanced_dataset:
        if args.balanced_output_dir is None:
            args.balanced_output_dir = args.data_dir / "balanced_19k"
        if args.balanced_output_csv is None:
            args.balanced_output_csv = args.balanced_output_dir / "labels.csv"

        balanced_df = build_balanced_augmented_dataset(
            df=df,
            data_dir=args.data_dir,
            output_dir=args.balanced_output_dir,
            output_csv=args.balanced_output_csv,
            target_distribution=TARGET_DISTRIBUTION,
            seed=args.seed,
        )

        print("\nBalanced dataset statistics:")
        print_statistics(balanced_df)
        print(
            "Use these for training:\n"
            f"  images_dir: {args.balanced_output_dir / 'images'}\n"
            f"  labels_csv: {args.balanced_output_csv}"
        )


if __name__ == "__main__":
    main()
