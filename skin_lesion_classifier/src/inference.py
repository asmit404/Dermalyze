"""
Inference Module for Skin Lesion Classification.

This module provides a clean interface for model inference, suitable for
integration with REST APIs and server-side deployment.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import (
    CLASS_LABELS,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    preprocess_image,
)
from src.models.efficientnet import SkinLesionClassifier


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
            model_size=model_config.get("size", "small"),
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
            "model_name": "EfficientNet-V2",
            "model_size": model_config.get("size", "small"),
            "num_classes": model_config.get("num_classes", 7),
            "image_size": self.image_size,
            "device": str(self.device),
            "checkpoint": str(self.checkpoint_path),
            "total_parameters": self.model.get_total_params(),
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


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test inference module")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    
    # Load predictor
    predictor = load_predictor(args.checkpoint)
    print(f"Model info: {predictor.get_model_info()}")
    
    # Make prediction
    result = predictor.predict(args.image)
    
    print(f"\nPrediction: {result['predicted_class']}")
    print(f"Description: {result['predicted_class_description']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nTop predictions:")
    for pred in result['top_k_predictions']:
        print(f"  {pred['class']}: {pred['probability']:.4f}")
    print(f"\n{result['disclaimer']}")
