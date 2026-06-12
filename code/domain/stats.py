from dataclasses import dataclass
from typing import List, Tuple, Literal

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
    possession_pct: PossessionPct
    distance_cm: List[DistanceCm]

