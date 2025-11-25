import cv2
import numpy as np
from typing import Optional, Tuple


class VideoCapture:
    def __init__(self, source: int = 0):
        self._source = source
        self._cap = None
        self._frame_size = (640, 480)

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._source)
        if self._cap.isOpened():
            self._frame_size = (
                int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            return True
        return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def frame_size(self) -> Tuple[int, int]:
        return self._frame_size

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()