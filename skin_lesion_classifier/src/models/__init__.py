"""Models package for skin lesion classification."""

from .efficientnet import (
    SkinLesionClassifier,
    FocalLoss,
    LabelSmoothingCrossEntropy,
    create_model,
    get_loss_function,
)
from .efficientnet_b1 import SkinLesionClassifierB1, create_model_b1
from .efficientnet_b2 import SkinLesionClassifierB2, create_model_b2
from .efficientnet_b3 import SkinLesionClassifierB3, create_model_b3
from .efficientnet_b4 import SkinLesionClassifierB4, create_model_b4

__all__ = [
    "SkinLesionClassifier",
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "create_model",
    "get_loss_function",
    "SkinLesionClassifierB1",
    "create_model_b1",
    "SkinLesionClassifierB2",
    "create_model_b2",
    "SkinLesionClassifierB3",
    "create_model_b3",
    "SkinLesionClassifierB4",
    "create_model_b4",
]
