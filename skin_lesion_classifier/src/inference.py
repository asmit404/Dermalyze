"""
Inference Module for Skin Lesion Classification.

This module provides a clean interface for model inference, suitable for
integration with REST APIs and server-side deployment.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data.dataset import (
    CLASS_LABELS,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_image,
)
from src.models.efficientnet import SkinLesionClassifier


def get_tta_transforms(image_size: int = 224) -> List[transforms.Compose]:
    """
    Get Test-Time Augmentation transforms.
    
    Returns a list of transforms that preserve semantic content:
    - Original
    - Horizontal flip
    - Vertical flip
    - Horizontal + Vertical flip
    - 90° rotation
    - 180° rotation
    - 270° rotation
    - Center crop with different scales
    
    Args:
        image_size: Target image size
        
    Returns:
        List of transform compositions
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    tta_transforms = [
        # Original
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            normalize,
        ]),
        # Vertical flip
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            normalize,
        ]),
        # Both flips
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            normalize,
        ]),
        # 90° rotation
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=(90, 90)),
            transforms.ToTensor(),
            normalize,
        ]),
        # 180° rotation
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=(180, 180)),
            transforms.ToTensor(),
            normalize,
        ]),
        # 270° rotation
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(degrees=(270, 270)),
            transforms.ToTensor(),
            normalize,
        ]),
        # Multi-scale crops
        transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]),
    ]
    
    return tta_transforms


class SkinLesionPredictor:
    """
    High-level inference class for skin lesion classification.
    
    This class handles model loading, image preprocessing, and prediction
    with a simple interface suitable for API integration.
    
    Example:
        >>> predictor = SkinLesionPredictor("checkpoint_best.pt")
        >>> result = predictor.predict("image.jpg")
        >>> print(result["predicted_class"], result["confidence"])
    """
    
    # Educational disclaimer
    DISCLAIMER = (
        "EDUCATIONAL USE ONLY: This system is for educational and research "
        "purposes only. It does not provide medical diagnosis or clinical "
        "decision-making. Always consult a qualified healthcare professional "
        "for medical advice."
    )
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
        image_size: int = 224,
    ):
        """
        Initialize the predictor.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda', 'cpu', or 'mps')
            image_size: Input image size (should match training)
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.image_size = image_size
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model, self.config = self._load_model()
        
        # Class information
        self.class_names = list(sorted(CLASS_LABELS.keys()))
        self.class_descriptions = CLASS_LABELS
    
    def _load_model(self) -> Tuple[SkinLesionClassifier, Dict[str, Any]]:
        """Load the model from checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        config = checkpoint.get("config", {})
        
        # Create model
        model_config = config.get("model", {})
        model = SkinLesionClassifier(
            num_classes=model_config.get("num_classes", 7),
            pretrained=False,
            dropout_rate=model_config.get("dropout_rate", 0.3),
        )
        
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()
        
        return model, config
    
    def preprocess(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, bytes],
    ) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: Image as path, PIL Image, numpy array, or bytes
            
        Returns:
            Preprocessed image tensor
        """
        # Handle bytes input (e.g., from API upload)
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        
        return preprocess_image(image, self.image_size).to(self.device)
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, bytes],
        top_k: int = 3,
        include_disclaimer: bool = True,
    ) -> Dict[str, Any]:
        """
        Make a prediction for a single image.
        
        Args:
            image: Input image (path, PIL Image, numpy array, or bytes)
            top_k: Number of top predictions to return
            include_disclaimer: Whether to include educational disclaimer
            
        Returns:
            Dictionary containing:
                - predicted_class: Most likely class label
                - predicted_class_description: Full description of predicted class
                - confidence: Confidence score for predicted class
                - probabilities: Dictionary of all class probabilities
                - top_k_predictions: List of top K predictions
                - disclaimer: Educational disclaimer (if requested)
        """
        # Preprocess
        tensor = self.preprocess(image)
        
        # Forward pass
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        
        # Get predictions
        probs_np = probs.cpu().numpy()
        predicted_idx = int(np.argmax(probs_np))
        predicted_class = IDX_TO_LABEL[predicted_idx]
        confidence = float(probs_np[predicted_idx])
        
        # All probabilities
        all_probs = {
            self.class_names[i]: float(probs_np[i])
            for i in range(len(self.class_names))
        }
        
        # Top K predictions
        top_k_indices = np.argsort(probs_np)[::-1][:top_k]
        top_k_predictions = [
            {
                "class": IDX_TO_LABEL[int(idx)],
                "description": CLASS_LABELS[IDX_TO_LABEL[int(idx)]],
                "probability": float(probs_np[idx]),
            }
            for idx in top_k_indices
        ]
        
        result = {
            "predicted_class": predicted_class,
            "predicted_class_description": CLASS_LABELS[predicted_class],
            "confidence": confidence,
            "probabilities": all_probs,
            "top_k_predictions": top_k_predictions,
        }
        
        if include_disclaimer:
            result["disclaimer"] = self.DISCLAIMER
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray, bytes]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple images.
        
        Args:
            images: List of input images
            top_k: Number of top predictions per image
            
        Returns:
            List of prediction dictionaries
        """
        # Preprocess all images
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0)
        
        # Forward pass
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1)
        
        # Process each prediction
        results = []
        for i in range(len(images)):
            probs_np = probs[i].cpu().numpy()
            predicted_idx = int(np.argmax(probs_np))
            predicted_class = IDX_TO_LABEL[predicted_idx]
            
            # Top K predictions
            top_k_indices = np.argsort(probs_np)[::-1][:top_k]
            top_k_predictions = [
                {
                    "class": IDX_TO_LABEL[int(idx)],
                    "description": CLASS_LABELS[IDX_TO_LABEL[int(idx)]],
                    "probability": float(probs_np[idx]),
                }
                for idx in top_k_indices
            ]
            
            results.append({
                "predicted_class": predicted_class,
                "predicted_class_description": CLASS_LABELS[predicted_class],
                "confidence": float(probs_np[predicted_idx]),
                "probabilities": {
                    self.class_names[j]: float(probs_np[j])
                    for j in range(len(self.class_names))
                },
                "top_k_predictions": top_k_predictions,
            })
        
        return results
    
    @torch.no_grad()
    def predict_with_tta(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, bytes],
        tta_mode: Literal["light", "medium", "full"] = "medium",
        aggregation: Literal["mean", "geometric_mean", "max"] = "mean",
        top_k: int = 3,
        include_disclaimer: bool = True,
    ) -> Dict[str, Any]:
        """
        Make prediction with Test-Time Augmentation.
        
        Args:
            image: Input image
            tta_mode: TTA complexity (light: 4 augs, medium: 8 augs, full: all)
            aggregation: How to aggregate predictions (mean, geometric_mean, max)
            top_k: Number of top predictions to return
            include_disclaimer: Whether to include educational disclaimer
            
        Returns:
            Prediction dictionary with TTA-enhanced probabilities
        """
        # Load image if needed
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        # Get TTA transforms based on mode
        all_transforms = get_tta_transforms(self.image_size)
        
        if tta_mode == "light":
            # Original + flips
            tta_transforms = all_transforms[:4]
        elif tta_mode == "medium":
            # Original + flips + rotations
            tta_transforms = all_transforms
        else:  # full
            tta_transforms = all_transforms
        
        # Collect predictions from all augmentations
        all_probs = []
        for transform in tta_transforms:
            tensor = transform(image).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            all_probs.append(probs.cpu().numpy())
        
        # Aggregate predictions
        all_probs = np.array(all_probs)  # Shape: (n_augmentations, n_classes)
        
        if aggregation == "mean":
            final_probs = np.mean(all_probs, axis=0)
        elif aggregation == "geometric_mean":
            # Geometric mean is better for probabilities
            final_probs = np.exp(np.mean(np.log(all_probs + 1e-10), axis=0))
            final_probs = final_probs / final_probs.sum()  # Normalize
        else:  # max
            final_probs = np.max(all_probs, axis=0)
        
        # Get predictions
        predicted_idx = int(np.argmax(final_probs))
        predicted_class = IDX_TO_LABEL[predicted_idx]
        confidence = float(final_probs[predicted_idx])
        
        # All probabilities
        all_probs_dict = {
            self.class_names[i]: float(final_probs[i])
            for i in range(len(self.class_names))
        }
        
        # Top K predictions
        top_k_indices = np.argsort(final_probs)[::-1][:top_k]
        top_k_predictions = [
            {
                "class": IDX_TO_LABEL[int(idx)],
                "description": CLASS_LABELS[IDX_TO_LABEL[int(idx)]],
                "probability": float(final_probs[idx]),
            }
            for idx in top_k_indices
        ]
        
        result = {
            "predicted_class": predicted_class,
            "predicted_class_description": CLASS_LABELS[predicted_class],
            "confidence": confidence,
            "probabilities": all_probs_dict,
            "top_k_predictions": top_k_predictions,
            "tta_mode": tta_mode,
            "tta_augmentations": len(tta_transforms),
            "aggregation_method": aggregation,
        }
        
        if include_disclaimer:
            result["disclaimer"] = self.DISCLAIMER
        
        return result
    
    def get_class_info(self) -> Dict[str, str]:
        """Get information about all classes."""
        return {
            "classes": self.class_descriptions.copy(),
            "num_classes": len(self.class_descriptions),
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        model_config = self.config.get("model", {})
        return {
            "model_name": "EfficientNet-V2-S",
            "num_classes": model_config.get("num_classes", 7),
            "image_size": self.image_size,
            "device": str(self.device),
            "checkpoint": str(self.checkpoint_path),
            "total_parameters": self.model.get_total_params(),
        }


class EnsemblePredictor:
    """
    Ensemble predictor that combines predictions from multiple models.
    
    Supports multiple ensemble strategies:
    - Mean averaging of probabilities
    - Weighted averaging with custom weights
    - Voting (hard or soft)
    - Can be combined with TTA for each model
    
    Example:
        >>> ensemble = EnsemblePredictor([
        ...     "model1_best.pt",
        ...     "model2_best.pt",
        ...     "model3_best.pt"
        ... ])
        >>> result = ensemble.predict("image.jpg", use_tta=True)
    """
    
    DISCLAIMER = SkinLesionPredictor.DISCLAIMER
    
    def __init__(
        self,
        checkpoint_paths: List[Union[str, Path]],
        weights: Optional[List[float]] = None,
        device: Optional[str] = None,
        image_size: int = 224,
    ):
        """
        Initialize ensemble predictor.
        
        Args:
            checkpoint_paths: List of paths to model checkpoints
            weights: Optional weights for each model (default: equal weights)
            device: Device for inference
            image_size: Input image size
        """
        if len(checkpoint_paths) == 0:
            raise ValueError("Must provide at least one checkpoint")
        
        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        self.image_size = image_size
        
        # Set weights
        if weights is None:
            self.weights = np.ones(len(checkpoint_paths)) / len(checkpoint_paths)
        else:
            if len(weights) != len(checkpoint_paths):
                raise ValueError("Number of weights must match number of checkpoints")
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()  # Normalize
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Load all models
        self.predictors = []
        for checkpoint_path in self.checkpoint_paths:
            predictor = SkinLesionPredictor(
                checkpoint_path=checkpoint_path,
                device=str(self.device),
                image_size=image_size,
            )
            self.predictors.append(predictor)
        
        # Class information (same for all models)
        self.class_names = self.predictors[0].class_names
        self.class_descriptions = self.predictors[0].class_descriptions
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, bytes],
        aggregation: Literal["mean", "weighted_mean", "geometric_mean", "max"] = "weighted_mean",
        use_tta: bool = False,
        tta_mode: Literal["light", "medium", "full"] = "medium",
        tta_aggregation: Literal["mean", "geometric_mean", "max"] = "mean",
        top_k: int = 3,
        include_disclaimer: bool = True,
    ) -> Dict[str, Any]:
        """
        Make ensemble prediction.
        
        Args:
            image: Input image
            aggregation: How to combine model predictions
            use_tta: Whether to use TTA for each model
            tta_mode: TTA complexity if use_tta=True
            tta_aggregation: How to aggregate TTA predictions
            top_k: Number of top predictions to return
            include_disclaimer: Whether to include disclaimer
            
        Returns:
            Ensemble prediction dictionary
        """
        # Collect predictions from all models
        all_model_probs = []
        
        for predictor in self.predictors:
            if use_tta:
                result = predictor.predict_with_tta(
                    image=image,
                    tta_mode=tta_mode,
                    aggregation=tta_aggregation,
                    include_disclaimer=False,
                )
            else:
                result = predictor.predict(image=image, include_disclaimer=False)
            
            # Extract probabilities as array
            probs = np.array([result["probabilities"][cls] for cls in self.class_names])
            all_model_probs.append(probs)
        
        all_model_probs = np.array(all_model_probs)  # Shape: (n_models, n_classes)
        
        # Aggregate model predictions
        if aggregation == "mean":
            final_probs = np.mean(all_model_probs, axis=0)
        elif aggregation == "weighted_mean":
            final_probs = np.average(all_model_probs, axis=0, weights=self.weights)
        elif aggregation == "geometric_mean":
            final_probs = np.exp(np.mean(np.log(all_model_probs + 1e-10), axis=0))
            final_probs = final_probs / final_probs.sum()
        else:  # max
            final_probs = np.max(all_model_probs, axis=0)
        
        # Get predictions
        predicted_idx = int(np.argmax(final_probs))
        predicted_class = IDX_TO_LABEL[predicted_idx]
        confidence = float(final_probs[predicted_idx])
        
        # All probabilities
        all_probs_dict = {
            self.class_names[i]: float(final_probs[i])
            for i in range(len(self.class_names))
        }
        
        # Top K predictions
        top_k_indices = np.argsort(final_probs)[::-1][:top_k]
        top_k_predictions = [
            {
                "class": IDX_TO_LABEL[int(idx)],
                "description": CLASS_LABELS[IDX_TO_LABEL[int(idx)]],
                "probability": float(final_probs[idx]),
            }
            for idx in top_k_indices
        ]
        
        # Calculate prediction variance across models (uncertainty measure)
        prediction_variance = np.var(all_model_probs, axis=0)
        prediction_std = np.std(all_model_probs, axis=0)
        
        result = {
            "predicted_class": predicted_class,
            "predicted_class_description": CLASS_LABELS[predicted_class],
            "confidence": confidence,
            "probabilities": all_probs_dict,
            "top_k_predictions": top_k_predictions,
            "ensemble_size": len(self.predictors),
            "aggregation_method": aggregation,
            "use_tta": use_tta,
            "prediction_uncertainty": float(prediction_std[predicted_idx]),  # Std of predicted class
            "mean_uncertainty": float(np.mean(prediction_std)),  # Mean std across all classes
        }
        
        if use_tta:
            result["tta_mode"] = tta_mode
            result["tta_aggregation"] = tta_aggregation
        
        if include_disclaimer:
            result["disclaimer"] = self.DISCLAIMER
        
        return result
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about the ensemble."""
        return {
            "num_models": len(self.predictors),
            "model_weights": self.weights.tolist(),
            "checkpoints": [str(p) for p in self.checkpoint_paths],
            "device": str(self.device),
            "image_size": self.image_size,
        }


def load_predictor(
    checkpoint_path: Union[str, Path],
    device: Optional[str] = None,
) -> SkinLesionPredictor:
    """
    Factory function to create a predictor instance.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device for inference
        
    Returns:
        Configured SkinLesionPredictor instance
    """
    return SkinLesionPredictor(checkpoint_path, device)


def load_ensemble_predictor(
    checkpoint_paths: List[Union[str, Path]],
    weights: Optional[List[float]] = None,
    device: Optional[str] = None,
) -> EnsemblePredictor:
    """
    Factory function to create an ensemble predictor instance.
    
    Args:
        checkpoint_paths: List of paths to model checkpoints
        weights: Optional weights for each model
        device: Device for inference
        
    Returns:
        Configured EnsemblePredictor instance
    """
    return EnsemblePredictor(checkpoint_paths, weights, device)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test inference module")
    parser.add_argument("--checkpoint", type=Path, help="Single model checkpoint")
    parser.add_argument("--ensemble", type=Path, nargs="+", help="Multiple checkpoints for ensemble")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--use-tta", action="store_true", help="Use test-time augmentation")
    parser.add_argument("--tta-mode", choices=["light", "medium", "full"], default="medium")
    args = parser.parse_args()
    
    if args.ensemble:
        # Ensemble prediction
        print(f"Loading ensemble with {len(args.ensemble)} models...")
        predictor = load_ensemble_predictor(args.ensemble)
        print(f"Ensemble info: {predictor.get_ensemble_info()}")
        
        result = predictor.predict(
            args.image,
            use_tta=args.use_tta,
            tta_mode=args.tta_mode,
        )
        print(f"\n[ENSEMBLE] Prediction: {result['predicted_class']}")
        print(f"Uncertainty: {result['prediction_uncertainty']:.4f}")
    else:
        # Single model prediction
        predictor = load_predictor(args.checkpoint)
        print(f"Model info: {predictor.get_model_info()}")
        
        if args.use_tta:
            result = predictor.predict_with_tta(
                args.image,
                tta_mode=args.tta_mode,
            )
            print(f"\n[TTA-{args.tta_mode.upper()}] Prediction: {result['predicted_class']}")
            print(f"Augmentations: {result['tta_augmentations']}")
        else:
            result = predictor.predict(args.image)
            print(f"\nPrediction: {result['predicted_class']}")
    
    print(f"Description: {result['predicted_class_description']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nTop predictions:")
    for pred in result['top_k_predictions']:
        print(f"  {pred['class']}: {pred['probability']:.4f}")
    
    if 'disclaimer' in result:
        print(f"\n{result['disclaimer']}")
