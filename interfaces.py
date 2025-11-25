from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> Optional[Any]:
        pass


class BaseClassifier(ABC):
    @abstractmethod
    def predict(self, data: Any) -> Any:
        pass


class BaseAnalyzer(ABC):
    @abstractmethod
    def analyze(self, data: Any) -> Any:
        pass