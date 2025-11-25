from dataclasses import dataclass
from typing import Optional, Dict
from drowsiness_analyzer import DrowsinessResult
from attention_analyzer import AttentionResult


@dataclass
class CombinedState:
    cognitive_state: str
    emotion: str
    confidence: float
    emotion_scores: Dict[str, float]
    drowsiness: Optional[DrowsinessResult]
    attention: Optional[AttentionResult]
    final_state: str
    face_detected: bool
    calibrating: bool = False


class StateAggregator:
    STATE_PRIORITY = {
        "durmiendo": 1,
        "no_mirando": 2,
        "frustrado": 3,
        "distraido": 4,
        "concentrado": 5,
        "entendiendo": 6
    }

    def aggregate(
        self,
        face_detected: bool,
        cognitive_state: str = "desconocido",
        emotion: str = "Unknown",
        confidence: float = 0.0,
        emotion_scores: Dict[str, float] = None,
        drowsiness: DrowsinessResult = None,
        attention: AttentionResult = None,
        calibrating: bool = False
    ) -> CombinedState:
        
        if not face_detected:
            return CombinedState(
                cognitive_state="desconocido",
                emotion="Unknown",
                confidence=0.0,
                emotion_scores=emotion_scores or {},
                drowsiness=None,
                attention=None,
                final_state="sin_rostro",
                face_detected=False,
                calibrating=False
            )
        
        final_state = cognitive_state
        
        if drowsiness and drowsiness.is_drowsy:
            final_state = "durmiendo"
        elif attention and not attention.is_looking_at_screen:
            final_state = "no_mirando"
        elif drowsiness and drowsiness.is_yawning:
            final_state = "distraido"
        
        return CombinedState(
            cognitive_state=cognitive_state,
            emotion=emotion,
            confidence=confidence,
            emotion_scores=emotion_scores or {},
            drowsiness=drowsiness,
            attention=attention,
            final_state=final_state,
            face_detected=True,
            calibrating=calibrating
        )