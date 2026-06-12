from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Robot:
    id: str
    team_id: str
    position: Tuple[float, float] = (0.0, 0.0)
    tracker_id: int = 0
    speed: float = 0.0
    angle: float = 0.0
    is_penalized: bool = False
    penalization_frames_left: int = 0

@dataclass
class Team:
    name: str
    color: str
    score: int = 0
    robots: List[Robot] = field(default_factory=list)

@dataclass
class Ball:
    id: str = "ball"
    position: Tuple[float, float] = (0.0, 0.0)
    tracker_id: Optional[int] = None
    speed_cm_s: float = 0.0
    direction_vector: Tuple[float, float] = (0.0, 0.0)

@dataclass
class FrameResult:
    frame_id: int
    robots: List[Robot]
    ball: Optional[Ball]

