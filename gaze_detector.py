import cv2
import numpy as np
from typing import Optional, Tuple


class GazeDetector:
    def __init__(self):
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.threshold = 70
        
    def detect_eyes(self, face_roi: np.ndarray) -> list:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        return eyes
    
    def detect_pupil(self, eye_roi: np.ndarray) -> Optional[Tuple[int, int]]:
        gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray_eye, (7, 7), 0)
        
        _, threshold_eye = cv2.threshold(
            blurred, self.threshold, 255, cv2.THRESH_BINARY_INV
        )
        
        kernel = np.ones((3, 3), np.uint8)
        threshold_eye = cv2.morphologyEx(threshold_eye, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(
            threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        valid_contours = [c for c in contours if cv2.contourArea(c) > 10]
        if not valid_contours:
            return None
        
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
        
        return None
    
    def is_looking_at_camera(self, eye_roi: np.ndarray) -> bool:
        pupil = self.detect_pupil(eye_roi)
        if pupil is None:
            return False
        
        h, w = eye_roi.shape[:2]
        cx, cy = pupil
        
        center_x = w / 2
        center_y = h / 2
        
        threshold_x = w * 0.3
        threshold_y = h * 0.3
        
        looking_center = (
            abs(cx - center_x) < threshold_x and
            abs(cy - center_y) < threshold_y
        )
        
        return looking_center
    
    def get_gaze_direction(self, eye_roi: np.ndarray) -> str:
        pupil = self.detect_pupil(eye_roi)
        if pupil is None:
            return "desconocido"
        
        h, w = eye_roi.shape[:2]
        cx, cy = pupil
        
        center_x = w / 2
        center_y = h / 2
        
        threshold = w * 0.2
        
        if abs(cx - center_x) < threshold and abs(cy - center_y) < threshold:
            return "centro"
        elif cx < center_x - threshold:
            return "izquierda"
        elif cx > center_x + threshold:
            return "derecha"
        elif cy < center_y - threshold:
            return "arriba"
        elif cy > center_y + threshold:
            return "abajo"
        
        return "centro"
    
    def analyze_gaze(self, face_roi: np.ndarray) -> Tuple[bool, str, int]:
        eyes = self.detect_eyes(face_roi)
        
        if len(eyes) == 0:
            return False, "sin_ojos", 0
        
        looking_count = 0
        directions = []
        
        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
            
            if self.is_looking_at_camera(eye_roi):
                looking_count += 1
            
            direction = self.get_gaze_direction(eye_roi)
            directions.append(direction)
        
        looking_at_camera = looking_count >= len(eyes) / 2
        
        main_direction = max(set(directions), key=directions.count) if directions else "desconocido"
        
        return looking_at_camera, main_direction, len(eyes)
    
    def draw_eye_analysis(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int], 
                         looking_at_camera: bool, gaze_direction: str, eye_count: int) -> np.ndarray:
        x, y, w, h = face_bbox
        result = frame.copy()
        
        face_roi = frame[y:y+h, x:x+w]
        eyes = self.detect_eyes(face_roi)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(result, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 255, 0), 2)
            
            eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
            pupil = self.detect_pupil(eye_roi)
            
            if pupil:
                px, py = pupil
                cv2.circle(result, (x+ex+px, y+ey+py), 3, (0, 255, 0), -1)
        
        status_text = "MIRANDO CAMARA" if looking_at_camera else "NO MIRANDO"
        status_color = (0, 255, 0) if looking_at_camera else (0, 0, 255)
        
        cv2.putText(result, status_text, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        cv2.putText(result, f"Direccion: {gaze_direction}", (x, y+h+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
