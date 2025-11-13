import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List


class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def detect(self, image: np.ndarray) -> Optional[np.ndarray]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None
    
    def get_bbox_from_landmarks(self, landmarks, image_shape) -> Tuple[int, int, int, int]:
        h, w = image_shape[:2]
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]
        
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))
        
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def crop_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        return image[y:y+h, x:x+w]
    
    def draw_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int], 
                  color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        x, y, w, h = bbox
        result = image.copy()
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
        return result
