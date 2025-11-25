import cv2
import numpy as np
from typing import Optional, Tuple
from interfaces import BaseDetector
from config import DetectorConfig


class FaceDetector(BaseDetector):
    def __init__(self, config: DetectorConfig = None):
        self._config = config or DetectorConfig()
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        
    def detect(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self._config.scale_factor,
            minNeighbors=self._config.min_neighbors,
            minSize=self._config.min_face_size
        )
        if len(faces) == 0:
            return None
        largest = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest)
    
    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        padding = int(self._config.face_padding * min(w, h))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        return image[y1:y2, x1:x2]