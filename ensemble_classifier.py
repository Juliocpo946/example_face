from hsemotion.facial_emotions import HSEmotionRecognizer
from deepface import DeepFace
from feat import Detector
import numpy as np
from typing import Tuple, Dict, List
from collections import deque


class EnsembleCognitiveClassifier:
    def __init__(self):
        print("[INFO] Inicializando Ensemble de 3 modelos...")
        print("[INFO] NOTA: Primera ejecución descargará modelos (~600MB total)")
        
        self.models_loaded = {'hsemotion': False, 'deepface': False, 'pyfeat': False}
        
        try:
            print("[INFO] Cargando HSEmotion...")
            self.hsemotion = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device='cpu')
            self.models_loaded['hsemotion'] = True
            print("[INFO] HSEmotion cargado")
        except Exception as e:
            print(f"[ERROR] HSEmotion falló: {str(e)}")
        
        try:
            print("[INFO] DeepFace listo (carga bajo demanda)")
            self.models_loaded['deepface'] = True
        except Exception as e:
            print(f"[ERROR] DeepFace falló: {str(e)}")
        
        try:
            print("[INFO] Cargando Py-Feat...")
            self.pyfeat = Detector(
                face_model='retinaface',
                landmark_model='mobilenet',
                au_model='xgb',
                emotion_model='resmasknet'
            )
            self.models_loaded['pyfeat'] = True
            print("[INFO] Py-Feat cargado")
        except Exception as e:
            print(f"[ERROR] Py-Feat falló: {str(e)}")
        
        if not any(self.models_loaded.values()):
            print("[ERROR] Ningún modelo se cargó correctamente")
        else:
            models_ok = [k for k, v in self.models_loaded.items() if v]
            print(f"[INFO] Modelos activos: {', '.join(models_ok)}")
        
        self.weights = {
            'hsemotion': 0.50,
            'deepface': 0.30,
            'pyfeat': 0.20
        }
        
        self.emotion_mapping = {
            'hsemotion': {
                'Anger': 'angry',
                'Contempt': 'disgust',
                'Disgust': 'disgust',
                'Fear': 'fear',
                'Happiness': 'happy',
                'Neutral': 'neutral',
                'Sadness': 'sad',
                'Surprise': 'surprise'
            },
            'deepface': {
                'angry': 'angry',
                'disgust': 'disgust',
                'fear': 'fear',
                'happy': 'happy',
                'sad': 'sad',
                'surprise': 'surprise',
                'neutral': 'neutral'
            },
            'pyfeat': {
                'anger': 'angry',
                'disgust': 'disgust',
                'fear': 'fear',
                'happiness': 'happy',
                'sadness': 'sad',
                'surprise': 'surprise',
                'neutral': 'neutral'
            }
        }
        
        self.emotion_to_cognitive = {
            'angry': 'frustrado',
            'disgust': 'frustrado',
            'sad': 'frustrado',
            'fear': 'distraido',
            'surprise': 'distraido',
            'happy': 'entendiendo',
            'neutral': 'concentrado'
        }
        
        self.window_size = 15
        self.emotion_history = deque(maxlen=self.window_size)
        self.confidence_history = deque(maxlen=self.window_size)
    
    def predict_hsemotion(self, face_crop: np.ndarray) -> Dict[str, float]:
        try:
            emotion, scores = self.hsemotion.predict_emotions(face_crop, logits=False)
            
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
            
            normalized = {}
            for orig_emotion, prob in emotion_dict.items():
                mapped_emotion = self.emotion_mapping['hsemotion'][orig_emotion]
                if mapped_emotion in normalized:
                    normalized[mapped_emotion] += prob
                else:
                    normalized[mapped_emotion] = prob
            
            return normalized
        except Exception as e:
            return {}
    
    def predict_deepface(self, face_crop: np.ndarray) -> Dict[str, float]:
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
            
            normalized = {self.emotion_mapping['deepface'][k]: v / 100.0 
                         for k, v in emotions.items()}
            
            return normalized
        except Exception as e:
            return {}
    
    def predict_pyfeat(self, face_crop: np.ndarray) -> Dict[str, float]:
        try:
            result = self.pyfeat.detect_image(face_crop)
            
            if result is None or len(result) == 0:
                return {}
            
            emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
            
            emotions = {}
            for col in emotion_cols:
                if col in result.columns:
                    emotions[col] = float(result[col].values[0])
            
            normalized = {self.emotion_mapping['pyfeat'][k]: v 
                         for k, v in emotions.items()}
            
            return normalized
        except Exception as e:
            return {}
    
    def weighted_ensemble(self, predictions: Dict[str, Dict[str, float]]) -> Tuple[str, float, Dict[str, float]]:
        emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        ensemble_probs = {emotion: 0.0 for emotion in emotion_labels}
        total_weight = 0.0
        
        for model_name, probs in predictions.items():
            if probs and model_name in self.weights:
                weight = self.weights[model_name]
                total_weight += weight
                
                for emotion in emotion_labels:
                    if emotion in probs:
                        ensemble_probs[emotion] += probs[emotion] * weight
        
        if total_weight > 0:
            for emotion in ensemble_probs:
                ensemble_probs[emotion] /= total_weight
        
        dominant_emotion = max(ensemble_probs, key=ensemble_probs.get)
        confidence = ensemble_probs[dominant_emotion]
        
        return dominant_emotion, confidence, ensemble_probs
    
    def temporal_smoothing(self, emotion: str, confidence: float) -> Tuple[str, float]:
        self.emotion_history.append(emotion)
        self.confidence_history.append(confidence)
        
        if len(self.emotion_history) < 3:
            return emotion, confidence
        
        emotion_counts = {}
        for e in self.emotion_history:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        
        smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
        
        smoothed_confidence = np.mean(list(self.confidence_history))
        
        return smoothed_emotion, smoothed_confidence
    
    def predict(self, face_crop: np.ndarray) -> Tuple[str, float, str, Dict]:
        if not any(self.models_loaded.values()):
            return 'desconocido', 0.0, '', {}
        
        predictions = {}
        
        if self.models_loaded['hsemotion']:
            predictions['hsemotion'] = self.predict_hsemotion(face_crop)
        
        if self.models_loaded['deepface']:
            predictions['deepface'] = self.predict_deepface(face_crop)
        
        if self.models_loaded['pyfeat']:
            predictions['pyfeat'] = self.predict_pyfeat(face_crop)
        
        if not predictions:
            return 'desconocido', 0.0, '', {}
        
        dominant_emotion, confidence, all_probs = self.weighted_ensemble(predictions)
        
        smoothed_emotion, smoothed_confidence = self.temporal_smoothing(dominant_emotion, confidence)
        
        cognitive_state = self.emotion_to_cognitive.get(smoothed_emotion, 'concentrado')
        
        all_probs_percent = {k: v * 100 for k, v in all_probs.items()}
        
        return cognitive_state, smoothed_confidence, smoothed_emotion, all_probs_percent
