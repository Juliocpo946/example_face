from deepface import DeepFace
import cv2
import numpy as np
from typing import Tuple, Dict


class DeepFaceCognitiveClassifier:
    def __init__(self):
        print("[INFO] Inicializando DeepFace...")
        try:
            self.model_loaded = True
            print("[INFO] DeepFace listo (modelos se descargan en primera ejecución)")
        except Exception as e:
            print(f"[ERROR] No se pudo inicializar DeepFace: {str(e)}")
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
    
    def predict(self, face_crop: np.ndarray) -> Tuple[str, float, str, Dict]:
        if not self.model_loaded:
            return 'desconocido', 0.0, '', {}
        
        try:
            result = DeepFace.analyze(
                img_path=face_crop,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result['emotion']
            dominant_emotion = result['dominant_emotion']
            confidence = emotions[dominant_emotion] / 100.0
            
            cognitive_state = self.emotion_to_cognitive.get(dominant_emotion, 'concentrado')
            
            return cognitive_state, confidence, dominant_emotion, emotions
            
        except Exception as e:
            print(f"[ERROR] Error en predicción DeepFace: {str(e)}")
            return 'desconocido', 0.0, '', {}
