from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image


class LocalNuisance(ABC):
    """Abstract base class for local nuisance generators."""

    @abstractmethod
    def apply(
        self,
        image: Image.Image,
        severity: int,
        rng: np.random.Generator,
        grad_map: Optional[np.ndarray] = None,
        max_iou: float = 0.0,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Apply nuisance to image at given severity.

        Returns:
            image_out: transformed image
            metadata: nuisance-specific metadata (for JSONL logging)
        """
        raise NotImplementedError
