import numpy as np
from scipy.spatial import distance
from typing import Tuple, Dict
from collections import deque


class DrowsinessDetector:
    def __init__(self):
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
        
        self.EAR_THRESHOLD = 0.23
        self.MAR_THRESHOLD = 0.65
        self.EYE_CLOSED_FRAMES = 60
        self.YAWN_FRAMES = 25
        
        self.eye_closed_counter = 0
        self.yawn_counter = 0
        self.total_blinks = 0
        self.total_yawns = 0
        
        self.blink_history = deque(maxlen=200)
        self.yawn_history = deque(maxlen=200)
        
        self.drowsiness_level = 0
        self.is_drowsy = False
        
    def calculate_ear(self, eye_points: np.ndarray) -> float:
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])
        C = distance.euclidean(eye_points[0], eye_points[3])
        
        if C == 0:
            return 0.3
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_mar(self, mouth_points: np.ndarray) -> float:
        A = distance.euclidean(mouth_points[1], mouth_points[7])
        B = distance.euclidean(mouth_points[2], mouth_points[6])
        C = distance.euclidean(mouth_points[0], mouth_points[4])
        
        if C == 0:
            return 0.3
        
        mar = (A + B) / (2.0 * C)
        return mar
    
    def get_landmarks_coords(self, landmarks, indices, image_shape):
        h, w = image_shape[:2]
        coords = []
        for idx in indices:
            landmark = landmarks.landmark[idx]
            coords.append([landmark.x * w, landmark.y * h])
        return np.array(coords)
    
    def detect(self, landmarks, image_shape) -> Tuple[bool, int, Dict]:
        if landmarks is None:
            return False, 0, {}
        
        left_eye = self.get_landmarks_coords(landmarks, self.LEFT_EYE_INDICES, image_shape)
        right_eye = self.get_landmarks_coords(landmarks, self.RIGHT_EYE_INDICES, image_shape)
        mouth = self.get_landmarks_coords(landmarks, self.MOUTH_INDICES, image_shape)
        
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        
        mar = self.calculate_mar(mouth)
        
        if ear < self.EAR_THRESHOLD:
            self.eye_closed_counter += 1
            if self.eye_closed_counter >= self.EYE_CLOSED_FRAMES:
                self.is_drowsy = True
                self.drowsiness_level = 3
        else:
            if self.eye_closed_counter >= 15:
                self.total_blinks += 1
                self.blink_history.append(1)
            self.eye_closed_counter = 0
        
        if mar > self.MAR_THRESHOLD:
            self.yawn_counter += 1
            if self.yawn_counter >= self.YAWN_FRAMES:
                self.total_yawns += 1
                self.yawn_history.append(1)
                self.yawn_counter = 0
        else:
            self.yawn_counter = 0
        
        recent_blinks = sum(self.blink_history)
        recent_yawns = sum(self.yawn_history)
        
        if not self.is_drowsy:
            if recent_yawns >= 4 or recent_blinks >= 35:
                self.drowsiness_level = 2
            elif recent_blinks >= 25:
                self.drowsiness_level = 1
            else:
                self.drowsiness_level = 0
        
        stats = {
            'ear': ear,
            'mar': mar,
            'blinks': self.total_blinks,
            'yawns': self.total_yawns,
            'eye_closed_frames': self.eye_closed_counter,
            'recent_blinks': recent_blinks,
            'recent_yawns': recent_yawns
        }
        
        return self.is_drowsy, self.drowsiness_level, stats
    
    def reset_drowsy_state(self):
        self.is_drowsy = False
        self.eye_closed_counter = 0
        if self.drowsiness_level == 3:
            self.drowsiness_level = 1