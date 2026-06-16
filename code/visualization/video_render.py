from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on runtime environment.
    raise ImportError(
        "visualization.video_render requires opencv-python and numpy."
    ) from exc

from visualization.tactical_map import TacticalMapRenderer

if TYPE_CHECKING:
    from domain.entities import Ball, FrameResult, Point2D, Robot
    from domain.events import FrameEvents
    from infra.event_bus import EventBus


Color = Tuple[int, int, int]


@dataclass(frozen=True)
class OverlayStyle:
    ally_bgr: Color = (216, 180, 0)
    rival_bgr: Color = (60, 35, 239)
    unknown_bgr: Color = (230, 230, 230)
    ball_bgr: Color = (0, 149, 255)
    text_bgr: Color = (255, 255, 255)
    event_bgr: Color = (0, 255, 255)
    robot_radius_px: int = 14
    ball_radius_px: int = 8


class VideoOverlayRenderer:
    """Draws existing FrameResult and FrameEvents data over video frames."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        *,
        style: OverlayStyle = OverlayStyle(),
        frame_event_type: str = "frame_result_raw",
        game_event_type: str = "frame_events",
        video_frame_event_type: str = "video_frame",
        output_event_type: str = "video_overlay",
        include_tactical_map: bool = False,
    ):
        self.style = style
        self.output_event_type = output_event_type
        self.include_tactical_map = include_tactical_map
        self._event_bus = event_bus
        self._latest_frame_result: Optional[FrameResult] = None
        self._latest_frame_events: Optional[FrameEvents] = None
        self._tactical_renderer = TacticalMapRenderer() if include_tactical_map else None

        if event_bus is not None:
            event_bus.subscribe(frame_event_type, self.on_frame_result)
            event_bus.subscribe(game_event_type, self.on_frame_events)
            event_bus.subscribe(video_frame_event_type, self.on_video_frame)

    def on_frame_result(self, frame_result: FrameResult) -> None:
        self._latest_frame_result = frame_result

    def on_frame_events(self, frame_events: FrameEvents) -> None:
        self._latest_frame_events = frame_events

    def on_video_frame(self, frame: "np.ndarray") -> None:
        if self._latest_frame_result is None:
            return
        rendered = self.render(frame, self._latest_frame_result, self._latest_frame_events)
        if self._event_bus is not None:
            self._event_bus.publish(self.output_event_type, rendered)

    def render(
        self,
        frame: "np.ndarray",
        frame_result: FrameResult,
        frame_events: Optional[FrameEvents] = None,
    ) -> "np.ndarray":
        output = frame.copy()
        for robot in frame_result.robots:
            self._draw_robot(output, robot)
        if frame_result.ball is not None:
            self._draw_ball(output, frame_result.ball)
        if frame_events is not None:
            self._draw_events(output, frame_events)

        cv2.putText(
            output,
            f"Frame {frame_result.frame_id}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.style.text_bgr,
            2,
            cv2.LINE_AA,
        )

        if self._tactical_renderer is not None:
            tactical = self._tactical_renderer.render(frame_result, frame_events)
            output = self._stack_side_by_side(output, tactical)
        return output

    def _draw_robot(self, frame: "np.ndarray", robot: Robot) -> None:
        point = self._point(robot.position_pixel)
        color = self._team_color(robot.team_id)
        cv2.circle(frame, point, self.style.robot_radius_px, color, 2, cv2.LINE_AA)
        cv2.putText(frame, robot.id, (point[0] + 16, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.style.text_bgr, 1, cv2.LINE_AA)
        if robot.is_penalized:
            cv2.putText(frame, "PEN", (point[0] - 13, point[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.style.event_bgr, 1, cv2.LINE_AA)

    def _draw_ball(self, frame: "np.ndarray", ball: Ball) -> None:
        point = self._point(ball.position_pixel)
        cv2.circle(frame, point, self.style.ball_radius_px, self.style.ball_bgr, -1, cv2.LINE_AA)
        cv2.circle(frame, point, self.style.ball_radius_px, self.style.text_bgr, 1, cv2.LINE_AA)

    def _draw_events(self, frame: "np.ndarray", frame_events: FrameEvents) -> None:
        y = 52
        for event in frame_events.eventos[-5:]:
            label = getattr(event, "type", event.__class__.__name__)
            cv2.putText(frame, str(label), (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.style.event_bgr, 1, cv2.LINE_AA)
            y += 20

    def _stack_side_by_side(self, frame: "np.ndarray", tactical: "np.ndarray") -> "np.ndarray":
        target_h = frame.shape[0]
        tactical_w = max(1, int(tactical.shape[1] * (target_h / tactical.shape[0])))
        tactical = cv2.resize(tactical, (tactical_w, target_h), interpolation=cv2.INTER_AREA)
        return np.hstack([frame, tactical])

    def _team_color(self, team_id: str) -> Color:
        normalized = team_id.lower()
        if normalized in {"ally", "allies", "azul", "blue"}:
            return self.style.ally_bgr
        if normalized in {"rival", "rivals", "rojo", "red"}:
            return self.style.rival_bgr
        return self.style.unknown_bgr
    
    @staticmethod
    def _point(p: "Point2D") -> Tuple[int, int]:
        return (int(round(p.x)), int(round(p.y)))


def render_video_overlay(
    frame: "np.ndarray",
    frame_result: FrameResult,
    frame_events: Optional[FrameEvents] = None,
    *,
    style: OverlayStyle = OverlayStyle(),
    include_tactical_map: bool = False,
) -> "np.ndarray":
    return VideoOverlayRenderer(
        style=style,
        include_tactical_map=include_tactical_map,
    ).render(frame, fram_result, frame_events)
