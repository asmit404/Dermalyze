"""
Data Preparation Script for HAM10000 Dataset.

This script prepares the HAM10000 dataset for training by:
1. Converting one-hot encoded ground_truth.csv to single-label format
2. Merging training labels with metadata.csv to create labels_with_metadata.csv
3. Validating the dataset structure
4. Generating dataset statistics

Expected dataset structure:
data/HAM10000_Training/
    images/
        ISIC_0024306.jpg
        ...
    ground_truth.csv (one-hot encoded: MEL, NV, BCC, AKIEC, BKL, DF, VASC)
    metadata.csv (contains age_approx, sex, anatom_site_general)

data/HAM10000_Val/
    images/
    ground_truth.csv

data/HAM10000_Test/
    images/
    ground_truth.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

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


def _convert_onehot_to_label(row: pd.Series) -> str:
    """Convert one-hot encoded labels to single label."""
    # HAM10000 label mapping
    label_map = {
        "MEL": "mel",
        "NV": "nv",
        "BCC": "bcc",
        "AKIEC": "akiec",
        "BKL": "bkl",
        "DF": "df",
        "VASC": "vasc",
    }

    # Find the label with value 1.0
    for col, label in label_map.items():
        if col in row.index and row[col] == 1.0:
            return label

    # If no label found, return unknown
    return "unknown"


def load_ground_truth(ground_truth_file: Path) -> pd.DataFrame:
    """
    Load HAM10000 ground_truth.csv (one-hot encoded) and convert to single-label format.

    Args:
        ground_truth_file: Path to ground_truth.csv

    Returns:
        DataFrame with columns: image_id, label
    """
    df = pd.read_csv(ground_truth_file)
    df.columns = [str(col).strip() for col in df.columns]

    # HAM10000 one-hot columns
    label_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    # Verify it's one-hot encoded
    if not all(col in df.columns for col in label_cols):
        raise ValueError(
            f"ground_truth.csv must contain one-hot encoded columns: {label_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    # Convert one-hot to single label
    df["label"] = df[label_cols].apply(_convert_onehot_to_label, axis=1)

    # Rename 'image' column to 'image_id' if needed
    if "image" in df.columns:
        df = df.rename(columns={"image": "image_id"})
    elif "image_id" not in df.columns:
        raise ValueError("ground_truth.csv must have 'image' or 'image_id' column")

    # Keep only image_id and label
    df = df[["image_id", "label"]].copy()

    return df


def merge_with_metadata(
    labels_df: pd.DataFrame,
    metadata_file: Path,
) -> pd.DataFrame:
    """
    Merge labels with metadata.csv.

    Args:
        labels_df: DataFrame with image_id and label columns
        metadata_file: Path to metadata.csv

    Returns:
        DataFrame with labels and metadata merged
    """
    metadata_df = pd.read_csv(metadata_file)
    metadata_df.columns = [str(col).strip() for col in metadata_df.columns]

    # Normalize image_id column name
    if "image" in metadata_df.columns and "image_id" not in metadata_df.columns:
        metadata_df = metadata_df.rename(columns={"image": "image_id"})
    elif "image_id" not in metadata_df.columns:
        raise ValueError("metadata.csv must have 'image' or 'image_id' column")

    # Build column mapping for metadata
    column_map = {}

    # Lesion ID column (critical for preventing data leakage)
    if "lesion_id" in metadata_df.columns:
        column_map["lesion_id"] = "lesion_id"

    # Age column
    if "age_approx" in metadata_df.columns:
        column_map["age_approx"] = "age_approx"
    elif "age" in metadata_df.columns:
        column_map["age"] = "age_approx"

    # Sex column
    if "sex" in metadata_df.columns:
        column_map["sex"] = "sex"

    # Anatomical site column
    if "anatom_site_general" in metadata_df.columns:
        column_map["anatom_site_general"] = "anatom_site"
    elif "anatom_site" in metadata_df.columns:
        column_map["anatom_site"] = "anatom_site"
    elif "localization" in metadata_df.columns:
        column_map["localization"] = "anatom_site"

    # Select columns to keep
    columns_to_keep = ["image_id"]
    rename_dict = {}

    for orig_col, target_col in column_map.items():
        if orig_col in metadata_df.columns:
            columns_to_keep.append(orig_col)
            if orig_col != target_col:
                rename_dict[orig_col] = target_col

    metadata_subset = metadata_df[columns_to_keep].copy()

    # Rename columns to standardized names
    if rename_dict:
        metadata_subset = metadata_subset.rename(columns=rename_dict)

    # Merge on image_id
    merged = labels_df.merge(metadata_subset, on="image_id", how="left")

    logger.info(f"Merged {len(merged)} images with metadata")
    if rename_dict:
        logger.info(f"Renamed columns: {rename_dict}")

    return merged


def prepare_dataset(
    data_dir: Path,
    output_csv: Path,
    validate_images: bool = True,
    include_metadata: bool = False,
) -> pd.DataFrame:
    """
    Prepare HAM10000 dataset for training.

    Args:
        data_dir: Root data directory (should contain 'images' and 'ground_truth.csv')
        output_csv: Path to save the prepared labels CSV
        validate_images: Whether to validate all images
        include_metadata: Whether to include metadata (only for training set)

    Returns:
        Prepared DataFrame
    """
    images_dir = data_dir / "images"
    ground_truth_file = data_dir / "ground_truth.csv"

    # Check directory structure
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not ground_truth_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_file}")

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = [f for f in images_dir.iterdir() if f.suffix in image_extensions]

    if not image_files:
        raise ValueError(f"No images found in {images_dir}")

    logger.info(f"Found {len(image_files)} images in {data_dir.name}")

    # Load ground truth
    logger.info(f"Loading ground truth from: {ground_truth_file}")
    df = load_ground_truth(ground_truth_file)
    logger.info(f"Loaded labels for {len(df)} images")

    # Merge with metadata if requested
    if include_metadata:
        metadata_file = data_dir / "metadata.csv"
        if metadata_file.exists():
            logger.info(f"Loading metadata from: {metadata_file}")
            df = merge_with_metadata(df, metadata_file)
        else:
            logger.warning(f"Metadata file not found: {metadata_file}")
            logger.warning("Proceeding without metadata")

    # Match images with labels
    image_ids_on_disk = {f.stem for f in image_files}
    image_ids_in_labels = set(df["image_id"])

    # Find mismatches
    missing_in_labels = image_ids_on_disk - image_ids_in_labels
    missing_on_disk = image_ids_in_labels - image_ids_on_disk

    if missing_in_labels:
        logger.warning(
            f"{len(missing_in_labels)} images on disk not in ground_truth.csv"
        )

    if missing_on_disk:
        logger.warning(
            f"{len(missing_on_disk)} images in ground_truth.csv not found on disk"
        )
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
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved labels to: {output_csv}")

    return df


def print_statistics(df: pd.DataFrame, dataset_name: str = "Dataset") -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} STATISTICS")
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
    if len(class_counts) > 1:
        max_class = class_counts.max()
        min_class = class_counts.min()
        imbalance_ratio = max_class / min_class
        print(f"\nClass imbalance ratio: {imbalance_ratio:.1f}:1")

    # Metadata columns info
    metadata_cols = [col for col in df.columns if col not in ["image_id", "label"]]
    if metadata_cols:
        print("\nMetadata columns:")
        print("-" * 40)
        for col in metadata_cols:
            non_null_count = df[col].notna().sum()
            non_null_pct = (non_null_count / len(df)) * 100
            unique_vals = df[col].nunique()
            print(
                f"  {col}: {non_null_count} non-null ({non_null_pct:.1f}%), {unique_vals} unique values"
            )

    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare HAM10000 dataset for training"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root data directory (e.g., data/HAM10000_Training)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output labels CSV path (default: {data_dir}/labels.csv or labels_with_metadata.csv)",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include metadata from metadata.csv (only for training set)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip image validation",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent

    if not args.data_dir.is_absolute():
        args.data_dir = (project_root / args.data_dir).resolve()

    # Set default output path
    if args.output is None:
        if args.include_metadata:
            args.output = args.data_dir / "labels_with_metadata.csv"
        else:
            args.output = args.data_dir / "labels.csv"
    elif not args.output.is_absolute():
        args.output = (project_root / args.output).resolve()

    # Prepare dataset
    df = prepare_dataset(
        data_dir=args.data_dir,
        output_csv=args.output,
        validate_images=not args.skip_validation,
        include_metadata=args.include_metadata,
    )

    # Print statistics
    dataset_name = args.data_dir.name
    print_statistics(df, dataset_name=dataset_name)


if __name__ == "__main__":
    main()
