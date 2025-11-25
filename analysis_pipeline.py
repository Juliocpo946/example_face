import numpy as np
from typing import Optional, Tuple
from config import AppConfig
from face_detector import FaceDetector
from landmark_extractor import LandmarkExtractor
from drowsiness_analyzer import DrowsinessAnalyzer
from attention_analyzer import AttentionAnalyzer
from emotion_classifier import EmotionClassifier
from state_aggregator import StateAggregator, CombinedState


class AnalysisPipeline:
    def __init__(self, config: AppConfig = None):
        self._config = config or AppConfig()
        self._face_detector = FaceDetector(self._config.detector)
        self._landmark_extractor = LandmarkExtractor()
        self._drowsiness_analyzer = DrowsinessAnalyzer(self._config.drowsiness)
        self._attention_analyzer = AttentionAnalyzer(self._config.attention)
        self._emotion_classifier = EmotionClassifier(self._config.emotion)
        self._state_aggregator = StateAggregator()
        self._last_state = None
        self._last_bbox = None

    def process(self, frame: np.ndarray) -> Tuple[CombinedState, Optional[Tuple[int, int, int, int]]]:
        bbox = self._face_detector.detect(frame)
        
        if bbox is None:
            state = self._state_aggregator.aggregate(face_detected=False)
            self._last_state = state
            self._last_bbox = None
            return state, None
        
        self._last_bbox = bbox
        
        landmarks = self._landmark_extractor.extract(frame)
        
        if landmarks is None:
            cognitive_state = "desconocido"
            emotion = "Unknown"
            confidence = 0.0
            emotion_scores = {}
            drowsiness = None
            attention = None
            calibrating = False
        else:
            drowsiness = self._drowsiness_analyzer.analyze(landmarks)
            attention = self._attention_analyzer.analyze(landmarks)
            calibrating = not self._attention_analyzer.is_calibrated
            
            face_crop = self._face_detector.crop_face(frame, bbox)
            cognitive_state, confidence, emotion, emotion_scores = self._emotion_classifier.predict(face_crop)
        
        state = self._state_aggregator.aggregate(
            face_detected=True,
            cognitive_state=cognitive_state,
            emotion=emotion,
            confidence=confidence,
            emotion_scores=emotion_scores,
            drowsiness=drowsiness,
            attention=attention,
            calibrating=calibrating
        )
        
        self._last_state = state
        return state, bbox

    def update_image_size(self, width: int, height: int):
        self._attention_analyzer.update_image_size(width, height)

    def reset(self):
        self._drowsiness_analyzer.reset()
        self._attention_analyzer.reset()
        self._emotion_classifier.reset()

    def reset_calibration(self):
        self._attention_analyzer.reset_calibration()

    def release(self):
        self._landmark_extractor.release()

    @property
    def last_state(self) -> Optional[CombinedState]:
        return self._last_state

    @property
    def last_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        return self._last_bbox