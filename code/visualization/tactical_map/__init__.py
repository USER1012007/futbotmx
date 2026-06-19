"""Mapa tactico de FutBotMX.

Fachada publica del renderer tactico. La cancha, entidades y eventos viven en
modulos internos para mantener separadas las responsabilidades de dibujo.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise ImportError("visualization.tactical_map requires numpy.") from exc

from visualization.tactical_map.entities import draw_ball, draw_frame_label, draw_robot, draw_trails
from visualization.tactical_map.events import TacticalEventOverlay
from visualization.tactical_map.field import draw_field
from visualization.tactical_map.style import FieldStyle

if TYPE_CHECKING:
    from domain.entities import Ball, FrameResult, Robot
    from domain.events import FrameEvents
    from infra.event_bus import EventBus


class TacticalMapRenderer:
    """Renderer del mapa tactico con soporte opcional de EventBus."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        *,
        style: FieldStyle = FieldStyle(),
        trail_length: int = 40,
        frame_event_type: str = "frame_result",
        game_event_type: str = "frame_events",
        output_event_type: str = "tactical_map",
    ):
        self.style = style
        self.output_event_type = output_event_type
        self._trails: Dict[str, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=trail_length))
        self._trail_expires: Dict[str, int] = {}
        self._visible_robots: Dict[str, Tuple[Robot, int]] = {}
        self._visible_ball: Optional[Tuple[Ball, int]] = None
        self._latest_events: Optional[FrameEvents] = None
        self._event_bus = event_bus
        self._event_overlay = TacticalEventOverlay(style)

        if event_bus is not None:
            event_bus.subscribe(frame_event_type, self.on_frame_result)
            event_bus.subscribe(game_event_type, self.on_frame_events)

    def on_frame_events(self, frame_events: "FrameEvents") -> None:
        self._latest_events = frame_events

    def on_frame_result(self, frame_result: "FrameResult") -> None:
        canvas = self.render(frame_result, self._latest_events)
        if self._event_bus is not None:
            self._event_bus.publish(self.output_event_type, canvas)

    def render(self, frame_result: "FrameResult", frame_events: Optional["FrameEvents"] = None) -> "np.ndarray":
        canvas, field_context = draw_field(self.style)

        self._remember_entities(frame_result)
        self._update_trails(frame_result)

        draw_trails(canvas, self.style, field_context, self._trails.values())

        for robot in self._robots_for_frame(frame_result.frame_id):
            if robot.position_metric is not None:
                draw_robot(canvas, self.style, field_context, robot)

        ball = self._ball_for_frame(frame_result.frame_id)
        if ball is not None and ball.position_metric is not None:
            draw_ball(canvas, self.style, field_context, ball)

        self._event_overlay.render(canvas, frame_events, frame_result.frame_id, frame_result, field_context)
        draw_frame_label(canvas, self.style, frame_result.frame_id)
        return canvas

    def draw_field(self) -> "np.ndarray":
        canvas, _ = draw_field(self.style)
        return canvas

    def reset(self) -> None:
        self._trails.clear()
        self._trail_expires.clear()
        self._visible_robots.clear()
        self._visible_ball = None
        self._latest_events = None
        self._event_overlay.reset()

    def _remember_entities(self, frame_result: "FrameResult") -> None:
        expires_at = frame_result.frame_id + self.style.entity_display_frames
        for robot in frame_result.robots:
            if robot.position_metric is not None:
                self._visible_robots[robot.id] = (robot, expires_at)
        if frame_result.ball is not None and frame_result.ball.position_metric is not None:
            self._visible_ball = (frame_result.ball, expires_at)

    def _robots_for_frame(self, frame_id: int) -> List["Robot"]:
        expired = [
            robot_id
            for robot_id, (_, expires_at) in self._visible_robots.items()
            if expires_at <= frame_id
        ]
        for robot_id in expired:
            self._visible_robots.pop(robot_id, None)
        return [robot for robot, _ in self._visible_robots.values()]

    def _ball_for_frame(self, frame_id: int) -> Optional["Ball"]:
        if self._visible_ball is None:
            return None
        ball, expires_at = self._visible_ball
        if expires_at <= frame_id:
            self._visible_ball = None
            return None
        return ball

    def _update_trails(self, frame_result: "FrameResult") -> None:
        for robot in frame_result.robots:
            if robot.position_metric is None:
                continue
            p = robot.position_metric
            self._trails[robot.id].append((p.x, p.y))
            self._trail_expires[robot.id] = frame_result.frame_id + self.style.entity_display_frames

        expired = [
            robot_id
            for robot_id, expires_at in self._trail_expires.items()
            if expires_at <= frame_result.frame_id
        ]
        for robot_id in expired:
            self._trail_expires.pop(robot_id, None)
            self._trails.pop(robot_id, None)


def draw_tactical_map(
    frame_result: "FrameResult",
    frame_events: Optional["FrameEvents"] = None,
    *,
    style: FieldStyle = FieldStyle(),
) -> "np.ndarray":
    return TacticalMapRenderer(style=style).render(frame_result, frame_events)
