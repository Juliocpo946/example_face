from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class DetectorConfig:
    min_face_size: Tuple[int, int] = (80, 80)
    scale_factor: float = 1.1
    min_neighbors: int = 5
    face_padding: float = 0.2


@dataclass
class DrowsinessConfig:
    ear_threshold: float = 0.22
    mar_threshold: float = 0.6
    drowsy_frames_threshold: int = 20
    yawn_frames_threshold: int = 15


@dataclass
class AttentionConfig:
    pitch_threshold: float = 45.0
    yaw_threshold: float = 45.0
    not_looking_frames_threshold: int = 25


@dataclass
class EmotionConfig:
    model_name: str = "enet_b0_8_best_afew"
    device: str = "cpu"
    history_size: int = 15
    min_history_for_smoothing: int = 3


@dataclass
class DisplayConfig:
    state_colors: Dict[str, Tuple[int, int, int]] = None
    font_scale: float = 0.7
    font_thickness: int = 2
    
    def __post_init__(self):
        if self.state_colors is None:
            self.state_colors = {
                "concentrado": (0, 255, 0),
                "distraido": (0, 165, 255),
                "frustrado": (0, 0, 255),
                "entendiendo": (255, 255, 0),
                "durmiendo": (128, 0, 128),
                "no_mirando": (100, 100, 100),
                "desconocido": (128, 128, 128)
            }


@dataclass
class AppConfig:
    detector: DetectorConfig = None
    drowsiness: DrowsinessConfig = None
    attention: AttentionConfig = None
    emotion: EmotionConfig = None
    display: DisplayConfig = None
    process_every_n_frames: int = 2
    
    def __post_init__(self):
        if self.detector is None:
            self.detector = DetectorConfig()
        if self.drowsiness is None:
            self.drowsiness = DrowsinessConfig()
        if self.attention is None:
            self.attention = AttentionConfig()
        if self.emotion is None:
            self.emotion = EmotionConfig()
        if self.display is None:
            self.display = DisplayConfig()