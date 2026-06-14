from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import domain.entities as domain_entities


try:
    Point2D = domain_entities.Point2D
except AttributeError:
    from dataclasses import dataclass

    @dataclass
    class Point2D:
        x: float
        y: float
        is_metric: bool = False

    domain_entities.Point2D = Point2D


try:
    Team = domain_entities.Team
except AttributeError:
    from dataclasses import dataclass, field
    from typing import List

    @dataclass
    class Team:
        name: str
        color: str
        score: int = 0
        robots: List["Robot"] = field(default_factory=list)

    domain_entities.Team = Team


from domain.entities import Ball, FrameResult, Robot, Team
from domain.events import CollisionEvent, FrameEvents, GoalEvent, PassEvent
from domain.stats import PossessionPct, Score, Statistics
from infra.configs import Config
from infra.event_bus import EventBus
from visualization.dashboard import DashboardRenderer
from visualization.layout import compose_final_frame
from visualization.tactical_map import TacticalMapRenderer
from visualization.video_render import VideoOverlayRenderer


FIELD_W_CM = 243.0
FIELD_H_CM = 182.0
VIDEO_W = 1280
VIDEO_H = 720


def main() -> None:
    output_dir = Config.OUTPUT_DIR / "mock_frames"
    output_dir.mkdir(parents=True, exist_ok=True)

    bus = EventBus()
    latest_outputs: Dict[str, Optional[np.ndarray]] = {
        "tactical_map": None,
        "video_overlay": None,
        "dashboard": None,
    }

    bus.subscribe("tactical_map", lambda image: _capture(latest_outputs, "tactical_map", image))
    bus.subscribe("video_overlay", lambda image: _capture(latest_outputs, "video_overlay", image))
    bus.subscribe("dashboard", lambda image: _capture(latest_outputs, "dashboard", image))

    TacticalMapRenderer(bus)
    VideoOverlayRenderer(bus, include_tactical_map=False)
    DashboardRenderer(bus)

    robots = _mock_robots()
    ball = _mock_ball(62.0, 82.0)
    events = _mock_events(robots)
    stats = Statistics(
        score=Score(allies=2, rivals=1),
        possession_pct=PossessionPct(allies=58.0, rivals=42.0),
        distance_cm={"allies": 1340.0, "rivals": 1210.0, "R1": 420.0, "R2": 365.0},
    )

    for frame_id in range(10):
        _update_positions(robots, ball, frame_id)
        frame_result = FrameResult(frame_id=frame_id, robots=robots, ball=ball)
        video_frame = _dummy_video_frame(frame_id)

        bus.publish("frame_result_raw", frame_result)
        bus.publish("frame_result", frame_result)
        bus.publish("frame_events", events)
        bus.publish("statistics", stats)
        bus.publish("video_frame", video_frame)

        tactical_map = _require_output(latest_outputs, "tactical_map")
        video_overlay = _require_output(latest_outputs, "video_overlay")
        dashboard = _require_output(latest_outputs, "dashboard")
        final_frame = compose_final_frame(video_overlay, tactical_map, dashboard)

        output_path = output_dir / f"frame_{frame_id:03d}.png"
        cv2.imwrite(str(output_path), final_frame)

    print(f"mock frames written to: {output_dir}")
    print(f"tactical_map: {latest_outputs['tactical_map'].shape}")
    print(f"video_overlay: {latest_outputs['video_overlay'].shape}")
    print(f"dashboard: {latest_outputs['dashboard'].shape}")
    print(f"final_frame: {final_frame.shape}")


def _capture(outputs: Dict[str, Optional[np.ndarray]], key: str, image: np.ndarray) -> None:
    outputs[key] = image


def _require_output(outputs: Dict[str, Optional[np.ndarray]], key: str) -> np.ndarray:
    image = outputs[key]
    if image is None:
        raise RuntimeError(f"Renderer did not publish '{key}'.")
    return image


def _mock_robots() -> list[Robot]:
    return [
        _mock_robot("R1", "allies", 44.0, 60.0, angle=0.0),
        _mock_robot("R2", "allies", 142.0, 84.0, angle=0.4, is_penalized=True),
        _mock_robot("R3", "rivals", 92.0, 128.0, angle=2.6),
        _mock_robot("R4", "rivals", 186.0, 116.0, angle=3.1),
    ]


def _mock_robot(
    robot_id: str,
    team_id: str,
    x_cm: float,
    y_cm: float,
    *,
    angle: float,
    is_penalized: bool = False,
) -> Robot:
    return Robot(
        id=robot_id,
        team_id=team_id,
        position_pixel=_metric_to_pixel(x_cm, y_cm),
        position_metric=Point2D(float(x_cm), float(y_cm), is_metric=True),
        angle=angle,
        is_penalized=is_penalized,
    )


def _mock_ball(x_cm: float, y_cm: float) -> Ball:
    ball = Ball(direction_vector=(0.8, -0.3), speed_cm_s=38.0)
    _set_metric_position(ball, x_cm, y_cm)
    return ball


def _mock_events(robots: list[Robot]) -> FrameEvents:
    allies = Team(name="allies", color="blue", score=2, robots=robots[:2])
    return FrameEvents(
        eventos=[
            PassEvent(type="pase", from_id="R1", to="R2", distance_cm=64.0),
            CollisionEvent(type="colision", robots=[robots[1], robots[2]], position=(240.0, 205.0)),
            GoalEvent(type="gol_valido", team=allies, velocity_cm_s=72.0, position=(18.0, 182.0)),
        ]
    )


def _update_positions(robots: list[Robot], ball: Ball, frame_id: int) -> None:
    for index, robot in enumerate(robots):
        dx = 2.5 + index * 0.6
        dy = (-1) ** index * 1.8
        p = robot.position_metric
        _set_metric_position(
            robot,
            _clamp(p.x + dx, 11.0, FIELD_W_CM - 11.0),
            _clamp(p.y + dy, 11.0, FIELD_H_CM - 11.0),
        )
        robot.angle += 0.08

    p = ball.position_metric
    _set_metric_position(
        ball,
        _clamp(p.x + 3.2, 4.0, FIELD_W_CM - 4.0),
        _clamp(p.y + np.sin(frame_id * 0.55) * 3.0, 4.0, FIELD_H_CM - 4.0),
    )
    ball.direction_vector = (0.9, float(np.cos(frame_id * 0.55) * 0.4))


def _dummy_video_frame(frame_id: int) -> np.ndarray:
    x_gradient = np.linspace(20, 120, VIDEO_W, dtype=np.uint8)
    y_gradient = np.linspace(30, 90, VIDEO_H, dtype=np.uint8)
    frame = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
    frame[:, :, 0] = x_gradient
    frame[:, :, 1] = y_gradient[:, None]
    frame[:, :, 2] = np.uint8(35 + frame_id * 12)

    cv2.rectangle(frame, (80, 70), (VIDEO_W - 80, VIDEO_H - 70), (80, 120, 80), 2)
    cv2.line(frame, (VIDEO_W // 2, 70), (VIDEO_W // 2, VIDEO_H - 70), (80, 120, 80), 1)
    cv2.putText(frame, "FutBotMX mock video", (90, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
    return frame


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return float(max(minimum, min(maximum, value)))


def _set_metric_position(obj: object, x_cm: float, y_cm: float) -> None:
    obj.position_metric = Point2D(float(x_cm), float(y_cm), is_metric=True)
    obj.position_pixel = _metric_to_pixel(x_cm, y_cm)


def _metric_to_pixel(x_cm: float, y_cm: float) -> Point2D:
    return Point2D(
        80.0 + x_cm / FIELD_W_CM * (VIDEO_W - 160.0),
        70.0 + y_cm / FIELD_H_CM * (VIDEO_H - 140.0),
        is_metric=False,
    )


if __name__ == "__main__":
    main()
