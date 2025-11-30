from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
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
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Apply nuisance to image at given severity.

        Returns:
            image_out: transformed image
            metadata: nuisance-specific metadata (for JSONL logging)
        """
        raise NotImplementedError
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
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
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Apply nuisance to image at given severity.

        Returns:
            image_out: transformed image
            metadata: nuisance-specific metadata (for JSONL logging)
        """
        raise NotImplementedError
