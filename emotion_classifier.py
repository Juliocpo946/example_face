import numpy as np
from typing import Tuple, Dict
from collections import deque
from hsemotion.facial_emotions import HSEmotionRecognizer
from interfaces import BaseClassifier
from config import EmotionConfig


class EmotionClassifier(BaseClassifier):
    EMOTION_TO_COGNITIVE = {
        "Anger": "frustrado",
        "Contempt": "frustrado",
        "Disgust": "frustrado",
        "Sadness": "frustrado",
        "Fear": "distraido",
        "Surprise": "distraido",
        "Happiness": "entendiendo",
        "Neutral": "concentrado"
    }

    def __init__(self, config: EmotionConfig = None):
        self._config = config or EmotionConfig()
        self._recognizer = HSEmotionRecognizer(
            model_name=self._config.model_name,
            device=self._config.device
        )
        self._emotion_history = deque(maxlen=self._config.history_size)
        self._confidence_history = deque(maxlen=self._config.history_size)

    def predict(self, face_crop: np.ndarray) -> Tuple[str, float, str, Dict[str, float]]:
        emotion, scores = self._recognizer.predict_emotions(face_crop, logits=False)
        
        emotion_dict = {
            "Anger": scores[0],
            "Contempt": scores[1],
            "Disgust": scores[2],
            "Fear": scores[3],
            "Happiness": scores[4],
            "Neutral": scores[5],
            "Sadness": scores[6],
            "Surprise": scores[7]
        }
        
        confidence = emotion_dict[emotion]
        self._emotion_history.append(emotion)
        self._confidence_history.append(confidence)
        
        if len(self._emotion_history) >= self._config.min_history_for_smoothing:
            emotion_counts = {}
            for e in self._emotion_history:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            emotion = max(emotion_counts, key=emotion_counts.get)
            confidence = np.mean(list(self._confidence_history))
        
        cognitive_state = self.EMOTION_TO_COGNITIVE.get(emotion, "concentrado")
        emotion_dict_percent = {k: v * 100 for k, v in emotion_dict.items()}
        
        return cognitive_state, confidence, emotion, emotion_dict_percent

    def reset(self):
        self._emotion_history.clear()
        self._confidence_history.clear()