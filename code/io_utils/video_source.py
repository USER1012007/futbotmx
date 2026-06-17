import cv2
from typing import Optional
from pathlib import Path

class VideoSource:
    def __init__(self, source: str | Path):
        self.source = str(source)
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise ValueError(f"No se pudo abrir la fuente de video: {self.source}")

    def get_frame(self) -> Optional[cv2.typing.MatLike]:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    @property
    def fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def release(self):
        self.cap.release()
