from fer import FER
import cv2
import numpy as np
from typing import Tuple


class FERCognitiveClassifier:
    def __init__(self):
        print("[INFO] Cargando modelo FER (CNN preentrenado)...")
        try:
            self.detector = FER(mtcnn=False)
            self.model_loaded = True
            print("[INFO] Modelo FER cargado exitosamente")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar FER: {str(e)}")
            self.model_loaded = False
        
        self.emotion_to_cognitive = {
            'angry': 'frustrado',
            'disgust': 'frustrado',
            'sad': 'frustrado',
            
            'fear': 'distraido',
            'surprise': 'distraido',
            
            'happy': 'entendiendo',
            
            'neutral': 'concentrado'
        }
    
    def predict(self, face_crop: np.ndarray) -> Tuple[str, float, str]:
        if not self.model_loaded:
            return 'desconocido', 0.0, ''
        
        try:
            emotions = self.detector.detect_emotions(face_crop)
            
            if not emotions or len(emotions) == 0:
                return 'concentrado', 0.5, 'neutral'
            
            emotion_scores = emotions[0]['emotions']
            
            top_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[top_emotion]
            
            cognitive_state = self.emotion_to_cognitive.get(top_emotion, 'concentrado')
            
            return cognitive_state, confidence, top_emotion
            
        except Exception as e:
            print(f"[ERROR] Error en predicción: {str(e)}")
            return 'desconocido', 0.0, ''