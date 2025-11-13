from hsemotion.facial_emotions import HSEmotionRecognizer
import numpy as np
from typing import Tuple, Dict
from collections import deque


class EmotionClassifier:
    def __init__(self):
        self.detector = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device='cpu')
        
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
        
        self.window_size = 15
        self.emotion_history = deque(maxlen=self.window_size)
        self.confidence_history = deque(maxlen=self.window_size)
    
    def predict(self, face_crop: np.ndarray) -> Tuple[str, float, str, Dict]:
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
        
        self.emotion_history.append(emotion)
        self.confidence_history.append(confidence)
        
        if len(self.emotion_history) >= 3:
            emotion_counts = {}
            for e in self.emotion_history:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            emotion = max(emotion_counts, key=emotion_counts.get)
            confidence = np.mean(list(self.confidence_history))
        
        cognitive_state = self.emotion_to_cognitive.get(emotion, 'concentrado')
        emotion_dict_percent = {k: v * 100 for k, v in emotion_dict.items()}
        
        return cognitive_state, confidence, emotion, emotion_dict_percent
