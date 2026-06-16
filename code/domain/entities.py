from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np


Team = Literal[
    "allies",
    "rivals",
    "ally",
    "rival",
    "azul",
    "rojo",
    "blue",
    "red",
    "unknown",
]


@dataclass
class Point2D:
    x: float
    y: float

    # False = pixeles de imagen/video
    # True = coordenadas métricas de cancha en centímetros
    is_metric: bool = False

    def to_tuple(self) -> Tuple[float, float]:
        return float(self.x), float(self.y)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "is_metric": bool(self.is_metric),
            "unit": "cm" if self.is_metric else "px",
        }


@dataclass
class Robot:
    id: str
    team_id: str
    position_pixel: Point2D
    position_metric: Optional[Point2D] = None
    tracker_id: int = 0
    speed: float = 0.0
    angle: float = 0.0
    is_penalized: bool = False
    penalization_frames_left: int = 0

    @property
    def team(self) -> str:
        return self.team_id

    @property
    def speed_cm_s(self) -> float:
        return self.speed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "team_id": self.team_id,
            "tracker_id": self.tracker_id,
            "position_pixel": point_to_dict(self.position_pixel),
            "position_metric": point_to_dict(self.position_metric),
            "speed_cm_s": float(self.speed),
            "angle": float(self.angle),
            "is_penalized": bool(self.is_penalized),
            "penalization_frames_left": int(self.penalization_frames_left),
        }


@dataclass
class Ball:
    id: str = "ball"
    position_pixel: Point2D = field(default_factory=lambda: Point2D(0.0, 0.0))
    position_metric: Optional[Point2D] = None

    tracker_id: Optional[int] = None
    speed_cm_s: float = 0.0

    direction_vector: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tracker_id": self.tracker_id,
            "position_pixel": point_to_dict(self.position_pixel),
            "position_metric": point_to_dict(self.position_metric),
            "speed_cm_s": float(self.speed_cm_s),
            "direction_vector": (
                float(self.direction_vector[0]),
                float(self.direction_vector[1]),
            ),
        }


@dataclass
class FrameResult:
    frame_id: int
    robots: List[Robot]
    ball: Optional[Ball]
    timestamp_s: Optional[float] = None
    field_mask: Optional[np.ndarray] = None
    repositions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": int(self.frame_id),
            "timestamp_s": float(self.timestamp_s) if self.timestamp_s is not None else None,
            "robots": [robot.to_dict() for robot in self.robots],
            "ball": self.ball.to_dict() if self.ball else None,
            "field_mask_present": self.field_mask is not None,
            "repositions": list(self.repositions),
        }


def point_to_dict(p: Optional[Point2D]) -> Optional[Dict[str, Any]]:
    return p.to_dict() if p is not None else None


def make_pixel_point(x: float, y: float) -> Point2D:
    return Point2D(float(x), float(y), is_metric=False)


def make_metric_point_cm(x: float, y: float) -> Point2D:
    return Point2D(float(x), float(y), is_metric=True)
