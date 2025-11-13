from hsemotion.facial_emotions import HSEmotionRecognizer
import numpy as np
from typing import Tuple, Dict


class HSEmotionCognitiveClassifier:
    def __init__(self, model_name: str = 'enet_b0_8_best_afew'):
        print(f"[INFO] Inicializando HSEmotion con modelo {model_name}...")
        try:
            self.detector = HSEmotionRecognizer(model_name=model_name, device='cpu')
            self.model_loaded = True
            print("[INFO] HSEmotion cargado exitosamente")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar HSEmotion: {str(e)}")
            self.model_loaded = False
        
        self.emotion_to_cognitive = {
            'Anger': 'frustrado',
            'Contempt': 'frustrado',
            'Disgust': 'frustrado',
            'Sadness': 'frustrado',
            
            'Fear': 'distraido',
            'Surprise': 'distraido',
            
            'Happiness': 'entendiendo',
            
            'Neutral': 'concentrado'
        }
    
    def predict(self, face_crop: np.ndarray) -> Tuple[str, float, str, Dict]:
        if not self.model_loaded:
            return 'desconocido', 0.0, '', {}
        
        try:
            emotion, scores = self.detector.predict_emotions(face_crop, logits=False)
            
            emotion_dict = {
                'Anger': scores[0],
                'Contempt': scores[1],
                'Disgust': scores[2],
                'Fear': scores[3],
                'Happiness': scores[4],
                'Neutral': scores[5],
                'Sadness': scores[6],
                'Surprise': scores[7]
            }
            
            confidence = emotion_dict[emotion]
            cognitive_state = self.emotion_to_cognitive.get(emotion, 'concentrado')
            
            emotion_dict_percent = {k: v * 100 for k, v in emotion_dict.items()}
            
            return cognitive_state, confidence, emotion, emotion_dict_percent
            
        except Exception as e:
            print(f"[ERROR] Error en predicción HSEmotion: {str(e)}")
            return 'desconocido', 0.0, '', {}
