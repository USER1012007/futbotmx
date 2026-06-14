from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Deque, Dict, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on runtime environment.
    raise ImportError(
        "visualization.tactical_map requires opencv-python and numpy."
    ) from exc

try:
    from domain.entities import Ball, FrameResult, Point2D, Robot
    from domain.events import FrameEvents
    from infra.event_bus import EventBus
except ImportError:
    code_dir = Path(__file__).resolve().parents[1]
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    from domain.entities import Ball, FrameResult, Point2D, Robot
    from domain.events import FrameEvents
    from infra.event_bus import EventBus


Color = Tuple[int, int, int]


@dataclass(frozen=True)
class FieldStyle:
    # Cancha
    field_width_px: int = 972   # 243 cm @ 4px/cm
    field_height_px: int = 728  # 182 cm @ 4px/cm
    margin_px: int = 30
    scale_px_cm: float = 4.0

    # Colores
    background_bgr: Color = (43, 67, 27)
    line_bgr: Color = (157, 198, 116)
    ally_bgr: Color = (216, 180, 0)
    rival_bgr: Color = (60, 35, 239)
    unknown_bgr: Color = (230, 230, 230)
    ball_bgr: Color = (0, 149, 255)
    text_bgr: Color = (255, 255, 255)
    event_bgr: Color = (0, 255, 255)
    trail_bgr: Color = (180, 180, 180)
    goal_ally_bgr: Color = (10, 214, 255)
    goal_rival_bgr: Color = (216, 180, 0)

    # Robot
    robot_radius_px: int = 22
    robot_direction_len_px: int = 36
    robot_label_offset_px: Tuple[int, int] = (24, 8)
    robot_label_scale: float = 0.7
    robot_outline_thickness_px: int = 1
    robot_circle_thickness_px: int = 2

    # Balón
    ball_radius_px: int = 14
    ball_direction_len_px: int = 40
    ball_outline_thickness_px: int = 1

    # Trails
    trail_thickness_px: int = 2

    # Eventos
    event_radius_px: int = 28
    event_label_scale: float = 0.7
    event_label_line_height_px: int = 28
    event_circle_thickness_px: int = 2
    event_display_frames: int = 30  # ~1s a 30fps
    event_max_visible: int = 4

    # Header (texto "Frame N")
    header_scale: float = 0.8
    header_offset_px: Tuple[int, int] = (0, 28)

    # Líneas de campo
    field_border_thickness_px: int = 2
    midline_thickness_px: int = 1
    center_circle_thickness_px: int = 1
    penalty_box_thickness_px: int = 1
    goal_thickness_px: int = 2

    # Arrow tip length (proporcion, no px)
    arrow_tip_length: float = 0.35

    @property
    def canvas_size(self) -> Tuple[int, int]:
        return (
            self.field_width_px + 2 * self.margin_px,
            self.field_height_px + 2 * self.margin_px,
        )


class TacticalMapRenderer:
    """Top-down field renderer fed by position_metric (cm) coordinates."""

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
        self._trails: Dict[str, Deque[Tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=trail_length)
        )
        self._latest_events: Optional[FrameEvents] = None
        self._event_bus = event_bus
        self._active_events: Deque[Tuple[object, int]] = deque()

        if event_bus is not None:
            event_bus.subscribe(frame_event_type, self.on_frame_result)
            event_bus.subscribe(game_event_type, self.on_frame_events)

    def on_frame_events(self, frame_events: FrameEvents) -> None:
        self._latest_events = frame_events

    def on_frame_result(self, frame_result: FrameResult) -> None:
        canvas = self.render(frame_result, self._latest_events)
        if self._event_bus is not None:
            self._event_bus.publish(self.output_event_type, canvas)

    def render(
        self,
        frame_result: FrameResult,
        frame_events: Optional[FrameEvents] = None,
    ) -> "np.ndarray":
        canvas = self.draw_field()

        for robot in frame_result.robots:
            if robot.position_metric is not None:
                self._append_trail(robot)

        self._draw_trails(canvas)

        for robot in frame_result.robots:
            if robot.position_metric is not None:
                self._draw_robot(canvas, robot)

        if frame_result.ball is not None and frame_result.ball.position_metric is not None:
            self._draw_ball(canvas, frame_result.ball)

        self._draw_events(canvas, frame_events, frame_result.frame_id)

        cv2.putText(
            canvas,
            f"Frame {frame_result.frame_id}",
            (self.style.margin_px + self.style.header_offset_px[0], self.style.header_offset_px[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.style.header_scale,
            self.style.text_bgr,
            1,
            cv2.LINE_AA,
        )
        return canvas

    def draw_field(self) -> "np.ndarray":
        canvas_w, canvas_h = self.style.canvas_size
        canvas = np.full(
            (canvas_h, canvas_w, 3),
            self.style.background_bgr,
            dtype=np.uint8,
        )
        m = self.style.margin_px
        w = self.style.field_width_px
        h = self.style.field_height_px
        s = self.style.scale_px_cm

        cv2.rectangle(canvas, (m, m), (m + w, m + h), self.style.line_bgr, self.style.field_border_thickness_px)

        # Línea de medio campo (vertical, separa izquierda/derecha)
        cv2.line(canvas, (m + w // 2, m), (m + w // 2, m + h), self.style.line_bgr, self.style.midline_thickness_px)

        # Círculo central, diámetro 60 cm -> radio 30 cm
        cv2.circle(canvas, (m + w // 2, m + h // 2), int(30 * s), self.style.line_bgr, self.style.center_circle_thickness_px)

        # Área penal: 80 cm (alto) x 100 cm (ancho desde la línea de fondo)
        penalty_h = int(80 * s)
        penalty_w = int(100 * s)
        penalty_y = m + (h - penalty_h) // 2

        cv2.rectangle(canvas, (m, penalty_y), (m + penalty_w, penalty_y + penalty_h), self.style.line_bgr, self.style.penalty_box_thickness_px)
        cv2.rectangle(canvas, (m + w - penalty_w, penalty_y), (m + w, penalty_y + penalty_h), self.style.line_bgr, self.style.penalty_box_thickness_px)

        # Portería: 60 cm (alto) x 5 cm (profundidad, hacia afuera de la línea de fondo)
        goal_h = int(60 * s)
        goal_w = int(5 * s)
        goal_y = m + (h - goal_h) // 2

        cv2.rectangle(canvas, (m - goal_w, goal_y), (m, goal_y + goal_h), self.style.goal_ally_bgr, self.style.goal_thickness_px)
        cv2.rectangle(canvas, (m + w, goal_y), (m + w + goal_w, goal_y + goal_h), self.style.goal_rival_bgr, self.style.goal_thickness_px)

        return canvas

    def _append_trail(self, robot: Robot) -> None:
        p = robot.position_metric
        self._trails[robot.id].append((p.x, p.y))

    def _draw_trails(self, canvas: "np.ndarray") -> None:
        for robot_id, points in self._trails.items():
            if len(points) < 2:
                continue
            pixel_points = [self._to_canvas_point(x, y) for x, y in points]
            for start, end in zip(pixel_points, pixel_points[1:]):
                cv2.line(canvas, start, end, self.style.trail_bgr, self.style.trail_thickness_px, cv2.LINE_AA)

    def _draw_robot(self, canvas: "np.ndarray", robot: Robot) -> None:
        p = robot.position_metric
        point = self._to_canvas_point(p.x, p.y)
        color = self._team_color(robot.team_id)
        radius = self.style.robot_radius_px
        thickness = self.style.robot_circle_thickness_px if not robot.is_penalized else -1
        cv2.circle(canvas, point, radius, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, point, radius, self.style.text_bgr, self.style.robot_outline_thickness_px, cv2.LINE_AA)
        if robot.angle:
            direction = (
                int(point[0] + np.cos(robot.angle) * self.style.robot_direction_len_px),
                int(point[1] + np.sin(robot.angle) * self.style.robot_direction_len_px),
            )
            cv2.arrowedLine(canvas, point, direction, self.style.text_bgr, 1, cv2.LINE_AA, tipLength=self.style.arrow_tip_length)
        offset_x, offset_y = self.style.robot_label_offset_px
        cv2.putText(canvas, robot.id, (point[0] + offset_x, point[1] + offset_y), cv2.FONT_HERSHEY_SIMPLEX, self.style.robot_label_scale, self.style.text_bgr, 1, cv2.LINE_AA)

    def _draw_ball(self, canvas: "np.ndarray", ball: Ball) -> None:
        p = ball.position_metric
        point = self._to_canvas_point(p.x, p.y)
        radius = self.style.ball_radius_px
        cv2.circle(canvas, point, radius, self.style.ball_bgr, -1, cv2.LINE_AA)
        cv2.circle(canvas, point, radius, self.style.text_bgr, self.style.ball_outline_thickness_px, cv2.LINE_AA)
        if ball.direction_vector != (0.0, 0.0):
            dx, dy = ball.direction_vector
            end = (
                int(point[0] + dx * self.style.ball_direction_len_px),
                int(point[1] + dy * self.style.ball_direction_len_px),
            )
            cv2.arrowedLine(canvas, point, end, self.style.ball_bgr, 1, cv2.LINE_AA, tipLength=self.style.arrow_tip_length)

    def _draw_events(self, canvas: "np.ndarray", frame_events: Optional[FrameEvents], frame_id: int) -> None:
        if frame_events is not None:
            for event in frame_events.eventos[-self.style.event_max_visible:]:
                self._active_events.append((event, frame_id + self.style.event_display_frames))

        while self._active_events and self._active_events[0][1] <= frame_id:
            self._active_events.popleft()

        y = self.style.margin_px + self.style.event_label_line_height_px
        seen_labels = []
        for event, _ in list(self._active_events)[-self.style.event_max_visible:]:
            label = getattr(event, "type", event.__class__.__name__)
            seen_labels.append(str(label))
            position = getattr(event, "position", None) or getattr(event, "last_position", None) or getattr(event, "target_position", None)
            if position is not None:
                cv2.circle(canvas, self._to_canvas_point(position[0], position[1]), self.style.event_radius_px, self.style.event_bgr, self.style.event_circle_thickness_px, cv2.LINE_AA)

        for label in seen_labels:
            cv2.putText(canvas, label, (self.style.margin_px + 6, y), cv2.FONT_HERSHEY_SIMPLEX, self.style.event_label_scale, (240, 240, 240), 1, cv2.LINE_AA)
            y += self.style.event_label_line_height_px

    def _to_canvas_point(self, x_cm: float, y_cm: float) -> Tuple[int, int]:
        m = self.style.margin_px
        s = self.style.scale_px_cm
        return (int(round(x_cm * s + m)), int(round(y_cm * s + m)))

    def _team_color(self, team_id: str) -> Color:
        normalized = team_id.lower()
        if normalized in {"ally", "allies", "azul", "blue"}:
            return self.style.ally_bgr
        if normalized in {"rival", "rivals", "rojo", "red"}:
            return self.style.rival_bgr
        return self.style.unknown_bgr


def draw_tactical_map(
    frame_result: FrameResult,
    frame_events: Optional[FrameEvents] = None,
    *,
    style: FieldStyle = FieldStyle(),
) -> "np.ndarray":
    return TacticalMapRenderer(style=style).render(frame_result, frame_events)
