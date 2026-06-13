from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass
class Point2D:
    x: float
    y: float
    is_metric: bool = False  # False: pixels, True: meters

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

@dataclass
class Ball:
    id: str = "ball"
    position_pixel: Point2D = field(default_factory=lambda: Point2D(0.0, 0.0))
    position_metric: Optional[Point2D] = None
    tracker_id: Optional[int] = None
    speed_cm_s: float = 0.0
    direction_vector: Tuple[float, float] = (0.0, 0.0)

@dataclass
class FrameResult:
    frame_id: int
    robots: List[Robot]
    ball: Optional[Ball]

    def to_dict(self):
        # Conversión recursiva para serialización y cast a float nativo
        def point_to_dict(p: Optional[Point2D]):
            return {"x": float(p.x), "y": float(p.y), "is_metric": p.is_metric} if p else None

        return {
            "frame_id": self.frame_id,
            "robots": [
                {
                    "id": r.id,
                    "tracker_id": r.tracker_id,
                    "position_pixel": point_to_dict(r.position_pixel),
                    "position_metric": point_to_dict(r.position_metric)
                } for r in self.robots
            ],
            "ball": {
                "id": self.ball.id,
                "tracker_id": self.ball.tracker_id,
                "position_pixel": point_to_dict(self.ball.position_pixel),
                "position_metric": point_to_dict(self.ball.position_metric)
            } if self.ball else None
        }

