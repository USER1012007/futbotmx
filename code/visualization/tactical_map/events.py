from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("visualization.tactical_map.events requires opencv-python.") from exc

from visualization.common.types import Color
from visualization.tactical_map.field import FieldRenderContext, scale_px, to_canvas_point, ui_scale
from visualization.tactical_map.style import EVENT_COLORS, EVENT_LABELS, FieldStyle

if TYPE_CHECKING:
    from domain.entities import FrameResult, Robot
    from domain.events import FrameEvents


class TacticalEventOverlay:
    def __init__(self, style: FieldStyle) -> None:
        self.style = style
        self._active_events: Deque[Tuple[tuple, object, int]] = deque()

    def reset(self) -> None:
        self._active_events.clear()

    def render(
        self,
        canvas: "object",
        frame_events: Optional["FrameEvents"],
        frame_id: int,
        frame_result: "FrameResult",
        context: FieldRenderContext,
    ) -> None:
        self._remember_active_events(frame_events, frame_id)
        robot_by_id = {
            robot.id: robot
            for robot in frame_result.robots
            if robot.position_metric is not None
        }

        labels = []
        for _, event, _ in list(self._active_events)[-self.style.event_max_visible:]:
            event_type = getattr(event, "type", event.__class__.__name__)
            color = self._event_color(event_type)

            if event_type == "pase":
                self._draw_pass_event(canvas, event, robot_by_id, color, context)
            else:
                position = self._event_position(event)
                if position is not None:
                    self._draw_event_marker(canvas, event_type, position, color, context)

            label = self._event_label(event)
            if label not in labels:
                labels.append(label)

        self._draw_event_labels(canvas, labels)

    def _remember_active_events(self, frame_events: Optional["FrameEvents"], frame_id: int) -> None:
        if frame_events is not None:
            for event in frame_events.eventos[-self.style.event_max_visible:]:
                key = self._event_key(event)
                already_active = any(active_key == key for active_key, _, _ in self._active_events)
                if not already_active:
                    self._active_events.append((key, event, frame_id + self.style.event_display_frames))

        while self._active_events and self._active_events[0][2] <= frame_id:
            self._active_events.popleft()

    def _draw_pass_event(
        self,
        canvas: "object",
        event: object,
        robot_by_id: Dict[str, "Robot"],
        color: Color,
        context: FieldRenderContext,
    ) -> None:
        from_id = getattr(event, "from_robot_id", getattr(event, "from_id", None))
        to_id = getattr(event, "to_robot_id", getattr(event, "to", None))
        if from_id not in robot_by_id or to_id not in robot_by_id:
            return

        p1 = robot_by_id[from_id].position_metric
        p2 = robot_by_id[to_id].position_metric
        cv2.arrowedLine(
            canvas,
            to_canvas_point(self.style, context, p1.x, p1.y),
            to_canvas_point(self.style, context, p2.x, p2.y),
            color,
            scale_px(self.style, 2),
            cv2.LINE_AA,
            tipLength=self.style.arrow_tip_length,
        )

    def _draw_event_marker(
        self,
        canvas: "object",
        event_type: str,
        position_cm: Tuple[float, float],
        color: Color,
        context: FieldRenderContext,
    ) -> None:
        point = to_canvas_point(self.style, context, position_cm[0], position_cm[1])
        radius = scale_px(self.style, self.style.event_radius_px)

        if event_type == "gol_valido":
            cv2.drawMarker(canvas, point, color, markerType=cv2.MARKER_STAR, markerSize=radius * 2, thickness=scale_px(self.style, 2), line_type=cv2.LINE_AA)
            return
        if event_type == "gol_invalido":
            cv2.drawMarker(canvas, point, color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=radius * 2, thickness=scale_px(self.style, 2), line_type=cv2.LINE_AA)
            return
        if event_type == "colision":
            cv2.circle(canvas, point, radius, color, scale_px(self.style, 2), cv2.LINE_AA)
            cv2.drawMarker(canvas, point, color, markerType=cv2.MARKER_TILTED_CROSS, markerSize=radius, thickness=scale_px(self.style, 2), line_type=cv2.LINE_AA)
            return

        cv2.circle(canvas, point, radius, color, scale_px(self.style, 2), cv2.LINE_AA)

    def _draw_event_labels(self, canvas: "object", labels: List[str]) -> None:
        y = self.style.margin_px + self.style.header_h_px + scale_px(self.style, 14)
        for label in labels:
            cv2.putText(
                canvas,
                label,
                (self.style.margin_px, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                ui_scale(self.style, self.style.event_label_scale),
                self.style.text_bgr,
                1,
                cv2.LINE_AA,
            )
            y += scale_px(self.style, self.style.event_label_line_height_px)

    def _event_position(self, event: object) -> Optional[Tuple[float, float]]:
        for attr in (
            "position_cm",
            "position",
            "last_position_cm",
            "last_position",
            "target_position_cm",
            "target_position",
            "to_position_cm",
        ):
            value = getattr(event, attr, None)
            if value is not None:
                return value
        return None

    def _event_key(self, event: object) -> tuple:
        event_type = getattr(event, "type", event.__class__.__name__)
        frame = getattr(event, "frame", None)
        position = self._event_position(event)
        robot = getattr(event, "robot", None)
        robot_id = getattr(robot, "id", None)
        return (event_type, frame, str(position), robot_id)

    def _event_label(self, event: object) -> str:
        event_type = getattr(event, "type", event.__class__.__name__)
        return EVENT_LABELS.get(event_type, event_type)

    def _event_color(self, event_type: str) -> Color:
        return EVENT_COLORS.get(event_type, self.style.event_bgr)
