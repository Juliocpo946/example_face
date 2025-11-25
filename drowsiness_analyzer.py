import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from interfaces import BaseAnalyzer
from config import DrowsinessConfig
from landmark_extractor import FaceLandmarks


@dataclass
class DrowsinessResult:
    ear: float
    mar: float
    is_drowsy: bool
    is_yawning: bool
    drowsy_frames: int
    yawn_frames: int


class DrowsinessAnalyzer(BaseAnalyzer):
    def __init__(self, config: DrowsinessConfig = None):
        self._config = config or DrowsinessConfig()
        self._drowsy_counter = 0
        self._yawn_counter = 0

    def _calculate_ear(self, eye_points: List[Tuple[float, float]]) -> float:
        p1, p2, p3, p4, p5, p6 = eye_points
        vertical_1 = np.linalg.norm(np.array(p2) - np.array(p6))
        vertical_2 = np.linalg.norm(np.array(p3) - np.array(p5))
        horizontal = np.linalg.norm(np.array(p1) - np.array(p4))
        if horizontal == 0:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def _calculate_mar(self, mouth_points: List[Tuple[float, float]]) -> float:
        if len(mouth_points) < 8:
            return 0.0
        p_top = np.array(mouth_points[2])
        p_bottom = np.array(mouth_points[3])
        p_left = np.array(mouth_points[0])
        p_right = np.array(mouth_points[1])
        vertical = np.linalg.norm(p_top - p_bottom)
        horizontal = np.linalg.norm(p_left - p_right)
        if horizontal == 0:
            return 0.0
        return vertical / horizontal

    def analyze(self, landmarks: FaceLandmarks) -> DrowsinessResult:
        left_ear = self._calculate_ear(landmarks.left_eye)
        right_ear = self._calculate_ear(landmarks.right_eye)
        ear = (left_ear + right_ear) / 2.0
        mar = self._calculate_mar(landmarks.mouth)
        
        if ear < self._config.ear_threshold:
            self._drowsy_counter += 1
        else:
            self._drowsy_counter = max(0, self._drowsy_counter - 1)
            
        if mar > self._config.mar_threshold:
            self._yawn_counter += 1
        else:
            self._yawn_counter = max(0, self._yawn_counter - 1)
        
        is_drowsy = self._drowsy_counter >= self._config.drowsy_frames_threshold
        is_yawning = self._yawn_counter >= self._config.yawn_frames_threshold
        
        return DrowsinessResult(
            ear=ear,
            mar=mar,
            is_drowsy=is_drowsy,
            is_yawning=is_yawning,
            drowsy_frames=self._drowsy_counter,
            yawn_frames=self._yawn_counter
        )

    def reset(self):
        self._drowsy_counter = 0
        self._yawn_counter = 0