from feat import Detector
import numpy as np
from typing import Tuple, Dict


class PyFeatCognitiveClassifier:
    def __init__(self):
        print("[INFO] Inicializando Py-Feat (puede tardar en primera ejecución)...")
        try:
            self.detector = Detector(
                face_model='retinaface',
                landmark_model='mobilenet',
                au_model='xgb',
                emotion_model='resmasknet'
            )
            self.model_loaded = True
            print("[INFO] Py-Feat cargado exitosamente")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar Py-Feat: {str(e)}")
            self.model_loaded = False
        
        self.emotion_to_cognitive = {
            'anger': 'frustrado',
            'disgust': 'frustrado',
            'sadness': 'frustrado',
            
            'fear': 'distraido',
            'surprise': 'distraido',
            
            'happiness': 'entendiendo',
            
            'neutral': 'concentrado'
        }
    
    def predict(self, face_crop: np.ndarray) -> Tuple[str, float, str, Dict]:
        if not self.model_loaded:
            return 'desconocido', 0.0, '', {}
        
        try:
            result = self.detector.detect_image(face_crop)
            
            if result is None or len(result) == 0:
                return 'concentrado', 0.5, 'neutral', {}
            
            emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
            
            emotions = {}
            for col in emotion_cols:
                if col in result.columns:
                    emotions[col] = float(result[col].values[0])
            
            if not emotions:
                return 'concentrado', 0.5, 'neutral', {}
            
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]
            
            cognitive_state = self.emotion_to_cognitive.get(dominant_emotion, 'concentrado')
            
            emotions_percent = {k: v * 100 for k, v in emotions.items()}
            
            return cognitive_state, confidence, dominant_emotion, emotions_percent
            
        except Exception as e:
            print(f"[ERROR] Error en predicción Py-Feat: {str(e)}")
            return 'desconocido', 0.0, '', {}
