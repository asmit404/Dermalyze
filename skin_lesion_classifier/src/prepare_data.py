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
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

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
            "image": "image_id",
            "dx": "label",
            "diagnosis": "label",
            "lesion_id": "lesion_id",
        }

        # Rename columns
        for old_name, new_name in column_mapping.items():
            if old_name in metadata.columns and old_name != new_name:
                metadata = metadata.rename(columns={old_name: new_name})

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
        "--skip-validation",
        action="store_true",
        help="Skip image validation",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.data_dir / "labels.csv"

    df = prepare_dataset(
        data_dir=args.data_dir,
        output_csv=args.output,
        metadata_file=args.metadata,
        validate_images=not args.skip_validation,
    )

    print_statistics(df)


if __name__ == "__main__":
    main()
