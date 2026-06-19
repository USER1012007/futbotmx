"""Overlay de video de FutBotMX.

Que hace: dibuja robots, balon, eventos recientes y frame_id sobre el video.
Flujo: recibe frame crudo + FrameResult + FrameEvents, pinta anotaciones en pixeles
y publica/devuelve el video overlay sin componer otras vistas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on runtime environment.
    raise ImportError(
        "visualization.video_render requires opencv-python and numpy."
    ) from exc

if TYPE_CHECKING:
    from domain.entities import Ball, FrameResult, Point2D, Robot
    from domain.events import FrameEvents
    from infra.event_bus import EventBus

from visualization.common.drawing import team_color
from visualization.common.types import Color


@dataclass(frozen=True)
class OverlayStyle:
    """Estilo del overlay sobre video.

    Agrupa colores y radios para robots y balon.
    Mantiene el render configurable sin tocar la logica.
    Usa coordenadas en pixeles del frame original.
    """

    ally_bgr: Color = (216, 180, 0)
    rival_bgr: Color = (60, 35, 239)
    unknown_bgr: Color = (230, 230, 230)
    ball_bgr: Color = (0, 149, 255)
    text_bgr: Color = (255, 255, 255)
    event_bgr: Color = (0, 255, 255)
    robot_radius_px: int = 14
    ball_radius_px: int = 8


class VideoOverlayRenderer:
    """Renderer de anotaciones sobre frames de video.

    Guarda el ultimo FrameResult y FrameEvents si usa EventBus.
    Dibuja entidades desde position_pixel.
    Publica el overlay final en el evento configurado.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        *,
        style: OverlayStyle = OverlayStyle(),
        frame_event_type: str = "frame_result_raw",
        game_event_type: str = "frame_events",
        video_frame_event_type: str = "video_frame",
        output_event_type: str = "video_overlay",
    ):
        self.style = style
        self.output_event_type = output_event_type
        self._event_bus = event_bus
        self._latest_frame_result: Optional[FrameResult] = None
        self._latest_frame_events: Optional[FrameEvents] = None

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

    def _team_color(self, team_id: str) -> Color:
        return team_color(team_id, self.style.ally_bgr, self.style.rival_bgr, self.style.unknown_bgr)
    
    @staticmethod
    def _point(p: "Point2D") -> Tuple[int, int]:
        return (int(round(p.x)), int(round(p.y)))


def render_video_overlay(
    frame: "np.ndarray",
    frame_result: FrameResult,
    frame_events: Optional[FrameEvents] = None,
    *,
    style: OverlayStyle = OverlayStyle(),
) -> "np.ndarray":
    return VideoOverlayRenderer(style=style).render(frame, frame_result, frame_events)
