from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Robot:
    id: str
    position: Tuple[float, float] = (0.0, 0.0)
    tracker_id: int = 0
    velocity: float = 0.0
    angulo: float = 0.0

@dataclass
class Team:
    name: str
    color: str
    robots: List[Robot]

@dataclass
class Ball:
    id: str = "ball_01"
    position: Tuple[float, float] = (0.0, 0.0)
    velocidad_cm_s: float = 0.0
    vector_direccion: Tuple[float, float] = (0.0, 0.0)
    
