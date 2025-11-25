import cv2
import numpy as np
from typing import Tuple, Optional
from config import DisplayConfig
from state_aggregator import CombinedState


class DisplayRenderer:
    STATE_LABELS = {
        "concentrado": "CONCENTRADO",
        "distraido": "DISTRAIDO",
        "frustrado": "FRUSTRADO",
        "entendiendo": "ENTENDIENDO",
        "durmiendo": "DURMIENDO",
        "no_mirando": "NO MIRA PANTALLA",
        "sin_rostro": "SIN ROSTRO",
        "desconocido": "DESCONOCIDO"
    }

    def __init__(self, config: DisplayConfig = None):
        self._config = config or DisplayConfig()
        self._show_details = False

    def toggle_details(self):
        self._show_details = not self._show_details

    def render(
        self,
        frame: np.ndarray,
        state: CombinedState,
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        display = frame.copy()
        
        if bbox and state.face_detected:
            color = self._config.state_colors.get(state.final_state, (255, 255, 255))
            x, y, w, h = bbox
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        
        y_offset = 30
        
        if not state.face_detected:
            cv2.putText(
                display, "Sin rostro detectado", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            return display
        
        color = self._config.state_colors.get(state.final_state, (255, 255, 255))
        label = self.STATE_LABELS.get(state.final_state, state.final_state.upper())
        
        cv2.putText(
            display, f"Estado: {label}", (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )
        y_offset += 35
        
        cv2.putText(
            display, f"Confianza: {state.confidence:.0%}", (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, self._config.font_scale, color, 2
        )
        y_offset += 30
        
        cv2.putText(
            display, f"Emocion: {state.emotion}", (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        y_offset += 25
        
        if state.drowsiness:
            ear_color = (0, 255, 0) if state.drowsiness.ear >= 0.22 else (0, 0, 255)
            cv2.putText(
                display, f"EAR: {state.drowsiness.ear:.2f}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 1
            )
            y_offset += 20
            
            if state.drowsiness.is_yawning:
                cv2.putText(
                    display, "BOSTEZANDO", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
                )
                y_offset += 25
        
        if state.attention:
            if not state.attention.is_looking_at_screen:
                cv2.putText(
                    display, "NO ESTA MIRANDO LA PANTALLA", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2
                )
                y_offset += 25
        
        if hasattr(state, 'calibrating') and state.calibrating:
            cv2.putText(
                display, "Calibrando... mire a la pantalla", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
            )
            y_offset += 25
        
        if self._show_details and state.emotion_scores:
            y_offset += 10
            cv2.putText(
                display, "--- Detalles ---", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1
            )
            y_offset += 20
            
            sorted_emotions = sorted(
                state.emotion_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for emo, score in sorted_emotions:
                cv2.putText(
                    display, f"{emo}: {score:.1f}%", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
                )
                y_offset += 18
        
        return display