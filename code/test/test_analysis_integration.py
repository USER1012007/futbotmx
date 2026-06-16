from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import domain.entities as domain_entities
from analysis.event_detector import EventDetector
from analysis.stats_engine import StatsEngine
from domain.entities import Ball, FrameResult, Robot
from domain.events import FrameEvents
from domain.stats import Statistics
from infra.configs import Config
from infra.event_bus import EventBus
from io_utils.tracking_io import TrackingIO


try:
    Point2D = domain_entities.Point2D
except AttributeError:

    @dataclass
    class Point2D:
        x: float
        y: float
        is_metric: bool = False


def test_analysis_pipeline_publishes_events_and_statistics_from_tracking_data() -> None:
    tracking_rows = _load_tracking_rows()
    positions = [_tracking_position(row) for row in tracking_rows[:3]]

    bus = EventBus()
    published_events: List[FrameEvents] = []
    published_statistics: List[Statistics] = []

    EventDetector(bus)
    StatsEngine(bus)
    bus.subscribe("frame_events", published_events.append)
    bus.subscribe("statistics", published_statistics.append)

    frame_0 = _frame_result(
        frame_id=0,
        robot_position=positions[0],
        ball_position=positions[0],
    )
    frame_1 = _frame_result(
        frame_id=1,
        robot_position=positions[1],
        ball_position=(120.0, 90.0),
    )
    frame_2 = _frame_result(
        frame_id=2,
        robot_position=positions[2],
        ball_position=(130.0, 90.0),
    )

    bus.publish("frame_result", frame_0)
    bus.publish("frame_result", frame_1)
    bus.publish("frame_result", frame_2)

    assert len(published_events) == 3
    assert len(published_statistics) == 3

    assert any(event.type == "posesion" for event in published_events[0].eventos)
    assert not any(event.type == "posesion" for event in published_events[1].eventos)

    latest_stats = published_statistics[-1]
    assert latest_stats.possession_pct.allies > 0.0
    assert latest_stats.possession_pct.rivals > 0.0
    assert latest_stats.distance_cm["robot_tracking"] > 0.0
    assert latest_stats.distance_cm["allies"] > 0.0


def _load_tracking_rows() -> List[Dict[str, Any]]:
    tracking_file = Config.TRACKING_DIR / "test_tracking.jsonl"
    rows = TrackingIO(tracking_file).read_all()
    assert len(rows) >= 3
    return rows


def _tracking_position(row: Dict[str, Any]) -> Tuple[float, float]:
    x, y = row["data"]["pos"]
    return float(x), float(y)


def _frame_result(
    *,
    frame_id: int,
    robot_position: Tuple[float, float],
    ball_position: Tuple[float, float],
) -> FrameResult:
    robot = Robot(
        id="robot_tracking",
        team_id="allies",
        position_pixel=_metric_to_pixel(robot_position),
        position_metric=_metric_to_point(robot_position),
        tracker_id=1,
    )
    ball = Ball(
        position_pixel=_metric_to_pixel(ball_position),
        position_metric=_metric_to_point(ball_position),
        tracker_id=2,
    )

    return FrameResult(frame_id=frame_id, robots=[robot], ball=ball)


def _metric_to_point(metric_position: Tuple[float, float]) -> Point2D:
    x_cm, y_cm = metric_position
    return Point2D(x_cm, y_cm, is_metric=True)


def _metric_to_pixel(metric_position: Tuple[float, float]) -> Point2D:
    x_cm, y_cm = metric_position
    return Point2D(x_cm * 4.0, y_cm * 4.0, is_metric=False)


if __name__ == "__main__":
    test_analysis_pipeline_publishes_events_and_statistics_from_tracking_data()
    print("Analysis integration test passed")
