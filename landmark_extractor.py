import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class FaceLandmarks:
    left_eye: List[Tuple[float, float]]
    right_eye: List[Tuple[float, float]]
    mouth: List[Tuple[float, float]]
    nose_tip: Tuple[float, float]
    chin: Tuple[float, float]
    left_eye_outer: Tuple[float, float]
    right_eye_outer: Tuple[float, float]
    all_landmarks: List[Tuple[float, float, float]]


class LandmarkExtractor:
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    MOUTH_INDICES = [61, 291, 0, 17, 405, 321, 375, 78, 191, 80, 81, 82]
    NOSE_TIP_INDEX = 1
    CHIN_INDEX = 152
    LEFT_EYE_OUTER_INDEX = 263
    RIGHT_EYE_OUTER_INDEX = 33

    def __init__(self):
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract(self, image: np.ndarray) -> Optional[FaceLandmarks]:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
            
        face_landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        def get_point(idx: int) -> Tuple[float, float]:
            lm = face_landmarks.landmark[idx]
            return (lm.x * w, lm.y * h)
        
        def get_point_3d(idx: int) -> Tuple[float, float, float]:
            lm = face_landmarks.landmark[idx]
            return (lm.x * w, lm.y * h, lm.z * w)
        
        left_eye = [get_point(i) for i in self.LEFT_EYE_INDICES]
        right_eye = [get_point(i) for i in self.RIGHT_EYE_INDICES]
        mouth = [get_point(i) for i in self.MOUTH_INDICES]
        
        all_landmarks = [get_point_3d(i) for i in range(468)]
        
        return FaceLandmarks(
            left_eye=left_eye,
            right_eye=right_eye,
            mouth=mouth,
            nose_tip=get_point(self.NOSE_TIP_INDEX),
            chin=get_point(self.CHIN_INDEX),
            left_eye_outer=get_point(self.LEFT_EYE_OUTER_INDEX),
            right_eye_outer=get_point(self.RIGHT_EYE_OUTER_INDEX),
            all_landmarks=all_landmarks
        )

    def release(self):
        self._face_mesh.close()