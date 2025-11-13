import cv2
import numpy as np
from typing import Optional, Tuple


class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.min_face_size = (80, 80)
    
    def detect(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=self.min_face_size
        )
        
        if len(faces) == 0:
            return None
        
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest_face)
    
    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        padding = int(0.2 * min(w, h))
        
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        return image[y1:y2, x1:x2]
    
    def draw_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        result = image.copy()
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return result
