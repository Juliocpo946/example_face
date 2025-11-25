import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from collections import deque
from interfaces import BaseAnalyzer
from config import AttentionConfig
from landmark_extractor import FaceLandmarks


@dataclass
class AttentionResult:
    pitch: float
    yaw: float
    roll: float
    is_looking_at_screen: bool
    not_looking_frames: int


class AttentionAnalyzer(BaseAnalyzer):
    def __init__(self, config: AttentionConfig = None, image_size: Tuple[int, int] = (640, 480)):
        self._config = config or AttentionConfig()
        self._not_looking_counter = 0
        self._image_size = image_size
        self._baseline_pitch: Optional[float] = None
        self._baseline_yaw: Optional[float] = None
        self._calibration_frames = deque(maxlen=30)
        self._is_calibrated = False
        self._pose_history = deque(maxlen=5)
        
    def _calculate_face_direction(self, landmarks: FaceLandmarks) -> Tuple[float, float]:
        nose = np.array(landmarks.nose_tip)
        left_eye = np.array(landmarks.left_eye_outer)
        right_eye = np.array(landmarks.right_eye_outer)
        chin = np.array(landmarks.chin)
        
        eye_center = (left_eye + right_eye) / 2
        face_width = np.linalg.norm(right_eye - left_eye)
        
        if face_width < 1:
            return 0.0, 0.0
        
        horizontal_offset = (nose[0] - eye_center[0]) / face_width
        yaw = horizontal_offset * 90
        
        face_height = np.linalg.norm(chin - eye_center)
        if face_height < 1:
            return 0.0, yaw
            
        vertical_offset = (nose[1] - eye_center[1]) / face_height
        pitch = (vertical_offset - 0.3) * 90
        
        return pitch, yaw

    def _smooth_pose(self, pitch: float, yaw: float) -> Tuple[float, float]:
        self._pose_history.append((pitch, yaw))
        
        if len(self._pose_history) < 2:
            return pitch, yaw
            
        pitches = [p[0] for p in self._pose_history]
        yaws = [p[1] for p in self._pose_history]
        
        return np.median(pitches), np.median(yaws)

    def _calibrate(self, pitch: float, yaw: float):
        self._calibration_frames.append((pitch, yaw))
        
        if len(self._calibration_frames) >= 30 and not self._is_calibrated:
            pitches = [p[0] for p in self._calibration_frames]
            yaws = [p[1] for p in self._calibration_frames]
            
            pitch_std = np.std(pitches)
            yaw_std = np.std(yaws)
            
            if pitch_std < 15 and yaw_std < 15:
                self._baseline_pitch = np.median(pitches)
                self._baseline_yaw = np.median(yaws)
                self._is_calibrated = True

    def analyze(self, landmarks: FaceLandmarks) -> AttentionResult:
        raw_pitch, raw_yaw = self._calculate_face_direction(landmarks)
        pitch, yaw = self._smooth_pose(raw_pitch, raw_yaw)
        
        if not self._is_calibrated:
            self._calibrate(pitch, yaw)
            return AttentionResult(
                pitch=pitch,
                yaw=yaw,
                roll=0.0,
                is_looking_at_screen=True,
                not_looking_frames=0
            )
        
        relative_pitch = pitch - self._baseline_pitch
        relative_yaw = yaw - self._baseline_yaw
        
        is_looking = (
            abs(relative_pitch) <= self._config.pitch_threshold and
            abs(relative_yaw) <= self._config.yaw_threshold
        )
        
        if not is_looking:
            self._not_looking_counter += 1
        else:
            self._not_looking_counter = max(0, self._not_looking_counter - 2)
            
        sustained_not_looking = self._not_looking_counter >= self._config.not_looking_frames_threshold
        
        return AttentionResult(
            pitch=relative_pitch,
            yaw=relative_yaw,
            roll=0.0,
            is_looking_at_screen=not sustained_not_looking,
            not_looking_frames=self._not_looking_counter
        )

    def reset(self):
        self._not_looking_counter = 0
        
    def reset_calibration(self):
        self._baseline_pitch = None
        self._baseline_yaw = None
        self._calibration_frames.clear()
        self._is_calibrated = False
        self._pose_history.clear()
        self._not_looking_counter = 0
        
    def update_image_size(self, width: int, height: int):
        self._image_size = (width, height)
        
    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated