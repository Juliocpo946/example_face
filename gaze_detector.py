import numpy as np
from typing import Tuple
from scipy.spatial import distance


class GazeDetector:
    def __init__(self, debug=False):
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS_INDICES = [474, 475, 476, 477]
        self.RIGHT_IRIS_INDICES = [469, 470, 471, 472]
        
        self.looking_threshold = 10.0
        self.debug = debug
        
    def get_eye_center(self, landmarks, eye_indices, image_shape):
        h, w = image_shape[:2]
        points = []
        for idx in eye_indices:
            landmark = landmarks.landmark[idx]
            points.append([landmark.x * w, landmark.y * h])
        points = np.array(points)
        return np.mean(points, axis=0)
    
    def get_iris_center(self, landmarks, iris_indices, image_shape):
        h, w = image_shape[:2]
        points = []
        for idx in iris_indices:
            landmark = landmarks.landmark[idx]
            points.append([landmark.x * w, landmark.y * h])
        points = np.array(points)
        return np.mean(points, axis=0)
    
    def calculate_gaze_ratio(self, eye_center, iris_center):
        if eye_center is None or iris_center is None:
            return 0.5
        
        distance_vec = iris_center - eye_center
        distance_norm = np.linalg.norm(distance_vec)
        
        if distance_norm < 1:
            return 0.5
        
        return distance_norm
    
    def is_looking_at_camera(self, landmarks, image_shape) -> Tuple[bool, float, float]:
        left_eye_center = self.get_eye_center(landmarks, self.LEFT_EYE_INDICES, image_shape)
        right_eye_center = self.get_eye_center(landmarks, self.RIGHT_EYE_INDICES, image_shape)
        
        left_iris_center = self.get_iris_center(landmarks, self.LEFT_IRIS_INDICES, image_shape)
        right_iris_center = self.get_iris_center(landmarks, self.RIGHT_IRIS_INDICES, image_shape)
        
        left_ratio = self.calculate_gaze_ratio(left_eye_center, left_iris_center)
        right_ratio = self.calculate_gaze_ratio(right_eye_center, right_iris_center)
        
        avg_ratio = (left_ratio + right_ratio) / 2.0
        
        if self.debug:
            print(f"[GAZE DEBUG] Left: {left_ratio:.2f}, Right: {right_ratio:.2f}, Avg: {avg_ratio:.2f}, Threshold: {self.looking_threshold}")
        
        looking = avg_ratio < self.looking_threshold
        
        return looking, left_ratio, right_ratio
    
    def analyze_gaze(self, landmarks, image_shape) -> Tuple[bool, str]:
        if landmarks is None:
            return False, "sin_deteccion"
        
        looking, left_ratio, right_ratio = self.is_looking_at_camera(landmarks, image_shape)
        
        if looking:
            direction = "centro"
        else:
            if left_ratio > 12 or right_ratio > 12:
                direction = "desviado"
            else:
                direction = "lateral"
        
        return looking, direction
    
    def calibrate(self, landmarks, image_shape, num_samples=30):
        print("\n[CALIBRACION] Mira directamente a la camara...")
        print(f"[CALIBRACION] Recolectando {num_samples} muestras...")
        
        ratios = []
        for i in range(num_samples):
            _, left, right = self.is_looking_at_camera(landmarks, image_shape)
            avg = (left + right) / 2.0
            ratios.append(avg)
            if (i + 1) % 10 == 0:
                print(f"[CALIBRACION] Muestra {i+1}/{num_samples}")
        
        max_ratio = max(ratios)
        avg_ratio = np.mean(ratios)
        
        recommended_threshold = max_ratio * 1.3
        
        print(f"\n[CALIBRACION] Resultados:")
        print(f"  Ratio promedio: {avg_ratio:.2f}")
        print(f"  Ratio maximo: {max_ratio:.2f}")
        print(f"  Threshold recomendado: {recommended_threshold:.2f}")
        print(f"\nAgregar en gaze_detector.py linea 12:")
        print(f"  self.looking_threshold = {recommended_threshold:.1f}")
        
        return recommended_threshold
