from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Score:
    allies: int = 0
    rivals: int = 0

@dataclass
class PossessionPct:
    allies: float = 0.0
    rivals: float = 0.0

@dataclass
class DistanceCm:
    allies_distance: float = 0.0
    rivals_distance: float = 0.0

@dataclass
class Statistics:
    score: Score
    possession_pct: PossessionPct
    distance_cm: Dict[str, float] = field(default_factory=dict)
    event_counts: Dict[str, int] = field(default_factory=dict)
    passes_attempted: int = 0
    passes_successful: int = 0
    collisions: int = 0
    invalid_goals: int = 0
    penalizations: int = 0
    stopped_frames_by_robot: Dict[str, int] = field(default_factory=dict)
    possession_frames_by_robot: Dict[str, int] = field(default_factory=dict)
    avg_speed_cm_s: Dict[str, float] = field(default_factory=dict)
    max_speed_cm_s: Dict[str, float] = field(default_factory=dict)
