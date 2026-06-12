from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    TRACKING_DIR: Path = DATA_DIR / "tracking"
    VIDEO_DIR: Path = DATA_DIR / "videos"
    OUTPUT_DIR: Path = DATA_DIR / "outputs"

    SAM_MODEL_NAME: str = "sam3.pt"
    DETECTION_THRESHOLD: float = 0.5

    CAMERA_RESOLUTION: tuple[int, int] = (1920, 1080)
    
    FPS_LIMIT: int = 30
