from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Robot:
    id: str
    position: Tuple[float, float] = (0.0, 0.0)
    tracker_id: int = 0
    speed: float = 0.0
    angle: float = 0.0

@dataclass
class Team:
    name: str
    color: str
    robots: List[Robot]

@dataclass
class Ball:
    id: str = "ball_01"
    position: Tuple[float, float] = (0.0, 0.0)
    speed_cm_s: float = 0.0
    direction_vector: Tuple[float, float] = (0.0, 0.0)
    
