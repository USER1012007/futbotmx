import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from domain.entities import Ball, FrameResult, Point2D, Robot

class TrackingIO:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def save_frame_data(self, frame_id: int, data: Dict[str, Any]):
        entry = {"frame_id": frame_id, "data": data}
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def read_all(self):
        results = []
        if not self.file_path.exists():
            return results
            
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    results.append(json.loads(stripped))
        return results

    def read_frame_results(self) -> List[FrameResult]:
        return [tracking_row_to_frame_result(row) for row in self.read_all()]


def tracking_row_to_frame_result(row: Dict[str, Any]) -> FrameResult:
    data = row.get("data", row)
    frame_id = int(data.get("frame_id", data.get("frame", row.get("frame_id", row.get("frame", 0)))))
    timestamp_s = data.get("timestamp_s", row.get("timestamp_s"))

    robots = [
        robot_from_tracking_dict(robot_data, index)
        for index, robot_data in enumerate(data.get("robots") or [])
    ]
    ball_data = data.get("ball", data.get("balon"))
    ball = ball_from_tracking_dict(ball_data) if ball_data is not None else None

    return FrameResult(
        frame_id=frame_id,
        timestamp_s=float(timestamp_s) if timestamp_s is not None else None,
        robots=robots,
        ball=ball,
        repositions=list(data.get("repositions", data.get("reposicionamientos", [])) or []),
    )


def robot_from_tracking_dict(robot_data: Dict[str, Any], index: int = 0) -> Robot:
    tracker_id = int(robot_data.get("tracker_id") or 0)
    return Robot(
        id=str(robot_data.get("id") or f"robot_{tracker_id or index}"),
        team_id=str(robot_data.get("team_id") or _team_id_for_robot(robot_data, index)),
        position_pixel=point_from_tracking_dict(robot_data.get("position_pixel"), is_metric=False),
        position_metric=optional_point_from_tracking_dict(robot_data.get("position_metric"), is_metric=True),
        tracker_id=tracker_id,
        speed=float(robot_data.get("speed_cm_s", robot_data.get("speed", 0.0)) or 0.0),
        angle=float(robot_data.get("angle", robot_data.get("angulo_deg", 0.0)) or 0.0),
        is_penalized=bool(robot_data.get("is_penalized", False)),
        penalization_frames_left=int(robot_data.get("penalization_frames_left", 0) or 0),
    )


def ball_from_tracking_dict(ball_data: Dict[str, Any]) -> Ball:
    direction = ball_data.get("direction_vector", ball_data.get("vector_direccion", (0.0, 0.0)))
    return Ball(
        id=str(ball_data.get("id") or "ball"),
        position_pixel=point_from_tracking_dict(ball_data.get("position_pixel"), is_metric=False),
        position_metric=optional_point_from_tracking_dict(ball_data.get("position_metric"), is_metric=True),
        tracker_id=ball_data.get("tracker_id"),
        speed_cm_s=float(ball_data.get("speed_cm_s", ball_data.get("velocidad_cm_s", 0.0)) or 0.0),
        direction_vector=(float(direction[0]), float(direction[1])) if len(direction) >= 2 else (0.0, 0.0),
    )


def point_from_tracking_dict(point_data: Optional[Dict[str, Any]], *, is_metric: bool) -> Point2D:
    if point_data is None:
        return Point2D(0.0, 0.0, is_metric=is_metric)
    return Point2D(float(point_data["x"]), float(point_data["y"]), is_metric=is_metric)


def optional_point_from_tracking_dict(point_data: Optional[Dict[str, Any]], *, is_metric: bool) -> Optional[Point2D]:
    if point_data is None:
        return None
    return point_from_tracking_dict(point_data, is_metric=is_metric)


def _team_id_for_robot(robot_data: Dict[str, Any], index: int) -> str:
    robot_id = str(robot_data.get("id") or "").lower()
    if robot_id.startswith("r") or "rival" in robot_id:
        return "rivals"
    if robot_id.startswith("a") or "ally" in robot_id or "aliado" in robot_id:
        return "allies"
    return "rivals" if index % 2 else "allies"
