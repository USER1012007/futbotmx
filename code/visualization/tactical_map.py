"""Mapa tactico de FutBotMX.

Que hace: dibuja cancha, robots, balon, trails y marcadores de eventos.
Flujo: recibe FrameResult y FrameEvents, actualiza trails, proyecta coordenadas
metricas a pixeles y devuelve un frame BGR listo para componer.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple
import math

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise ImportError("visualization.tactical_map requires opencv-python and numpy.") from exc

if TYPE_CHECKING:
    from domain.entities import Ball, FrameResult, Robot
    from domain.events import FrameEvents
    from infra.event_bus import EventBus

from domain.field import FIELD_GEOMETRY, FieldGeometry as DomainFieldGeometry


Color = Tuple[int, int, int]


EVENT_LABELS: Dict[str, str] = {
    "gol_valido": "gol valido",
    "gol_invalido": "gol invalido",
    "pase": "pase",
    "colision": "colision",
    "posesion": "posesion",
    "fuera_de_cancha": "fuera",
    "robot_detenido": "detenido",
    "reposicion_balon": "rep. balon",
    "reposicion_robot": "rep. robot",
    "sacar_robot": "sacar robot",
    "panic": "panic",
}

EVENT_COLORS: Dict[str, Color] = {
    "gol_valido": (80, 220, 80),
    "gol_invalido": (40, 40, 255),
    "pase": (210, 120, 255),
    "colision": (40, 40, 255),
    "posesion": (0, 220, 255),
    "fuera_de_cancha": (0, 140, 255),
    "robot_detenido": (150, 150, 150),
    "reposicion_balon": (255, 180, 60),
    "reposicion_robot": (255, 130, 40),
    "sacar_robot": (40, 40, 200),
    "panic": (255, 60, 255),
}


@dataclass(frozen=True)
class FieldGeometry:
    """Medidas fisicas de la cancha en centimetros.

    Define dimensiones externas, lineas internas, areas y porterias.
    Expone propiedades derivadas para centro y rectangulo interior.
    Se usa como base para convertir centimetros a pixeles.
    """
    # Medidas físicas reales en cm
    outer_width_cm: float = 243.0
    outer_height_cm: float = 182.0

    # Rectángulo blanco interior: 219 x 158 cm: inset de 12 cm por lado
    boundary_inset_cm: float = 12.0

    # Círculo central: diámetro 60 cm
    center_circle_diameter_cm: float = 60.0
    center_spot_radius_cm: float = 2.0

    # Áreas laterales blancas
    penalty_box_depth_cm: float = 25.0
    penalty_box_height_cm: float = 80.0
    penalty_box_corner_radius_cm: float = 6.0

    # Porterías
    goal_depth_cm: float = 10.0
    goal_height_cm: float = 60.0

    @property
    def inner_width_cm(self) -> float:
        return self.outer_width_cm - 2.0 * self.boundary_inset_cm

    @property
    def inner_height_cm(self) -> float:
        return self.outer_height_cm - 2.0 * self.boundary_inset_cm

    @property
    def center_x_cm(self) -> float:
        return self.outer_width_cm / 2.0

    @property
    def center_y_cm(self) -> float:
        return self.outer_height_cm / 2.0


@dataclass(frozen=True)
class FieldStyle:
    """Estilo visual y escalas del mapa tactico.

    Agrupa colores, tamaños, grosores y parametros de eventos.
    Mantiene configurable el render sin cambiar la logica de dibujo.
    Sus valores se escalan segun el tamano de salida.
    """
    geometry: DomainFieldGeometry = FIELD_GEOMETRY
    mirror_x: bool = False

    # Canvas de salida
    output_size: Tuple[int, int] = (1280, 720)
    margin_px: int = 12

    # Colores
    wall_bgr: Color = (0, 0, 0)
    field_bgr: Color = (119, 179, 71)
    line_bgr: Color = (245, 245, 245)
    center_mark_bgr: Color = (0, 0, 0)
    ally_goal_bgr: Color = (0, 200, 255)
    rival_goal_bgr: Color = (220, 110, 60)
    ally_bgr: Color = (216, 180, 0)
    rival_bgr: Color = (60, 35, 239)
    unknown_bgr: Color = (230, 230, 230)
    ball_bgr: Color = (0, 149, 255)
    text_bgr: Color = (255, 255, 255)
    event_bgr: Color = (0, 255, 255)
    trail_bgr: Color = (185, 185, 185)

    # Escalado UI relativo
    base_ref_h: int = 720
    outer_wall_thickness_px: int = 8
    inner_line_thickness_px: int = 5
    center_line_thickness_px: int = 3
    circle_thickness_px: int = 5
    goal_thickness_px: int = 4

    # Robot
    robot_radius_px: int = 18
    robot_direction_len_px: int = 32
    robot_label_scale: float = 0.8
    robot_label_offset_px: Tuple[int, int] = (18, 7)
    robot_outline_thickness_px: int = 1
    robot_circle_thickness_px: int = 2

    # Balón
    ball_radius_px: int = 10
    ball_direction_len_px: int = 26
    ball_outline_thickness_px: int = 1

    # Trails
    trail_thickness_px: int = 2

    # Eventos/header
    event_radius_px: int = 24
    event_label_scale: float = 0.8
    event_label_line_height_px: int = 26
    event_display_frames: int = 75
    event_max_visible: int = 4
    entity_display_frames: int = 12
    header_scale: float = 0.8
    header_h_px: int = 24
    arrow_tip_length: float = 0.30


class TacticalMapRenderer:
    """Renderer del mapa tactico con soporte opcional de EventBus.

    Escucha frames y eventos, mantiene trails por robot y eventos activos.
    Dibuja cancha, entidades y overlays de eventos por frame.
    Publica el mapa renderizado si esta conectado al bus.
    """
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
        self._latest_events: Optional[FrameEvents] = None
        self._event_bus = event_bus
        self._active_events: Deque[Tuple[tuple, object, int]] = deque()
        self._visible_robots: Dict[str, Tuple[Robot, int]] = {}
        self._visible_ball: Optional[Tuple[Ball, int]] = None

        if event_bus is not None:
            event_bus.subscribe(frame_event_type, self.on_frame_result)
            event_bus.subscribe(game_event_type, self.on_frame_events)

    def on_frame_events(self, frame_events: FrameEvents) -> None:
        self._latest_events = frame_events

    def on_frame_result(self, frame_result: FrameResult) -> None:
        canvas = self.render(frame_result, self._latest_events)
        if self._event_bus is not None:
            self._event_bus.publish(self.output_event_type, canvas)

    def render(self, frame_result: FrameResult, frame_events: Optional[FrameEvents] = None) -> "np.ndarray":
        canvas = self.draw_field()
        self._remember_entities(frame_result)
        robots = self._robots_for_frame(frame_result.frame_id)
        ball = self._ball_for_frame(frame_result.frame_id)

        for robot in robots:
            if robot.position_metric is not None:
                self._append_trail(robot)

        self._draw_trails(canvas)

        for robot in robots:
            if robot.position_metric is not None:
                self._draw_robot(canvas, robot)

        if ball is not None and ball.position_metric is not None:
            self._draw_ball(canvas, ball)

        self._draw_events(canvas, frame_events, frame_result.frame_id, frame_result)

        cv2.putText(
            canvas,
            f"Frame {frame_result.frame_id}",
            (max(12, self.style.margin_px), max(20, self.style.margin_px + 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._ui_scale(self.style.header_scale),
            self.style.text_bgr,
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _remember_entities(self, frame_result: FrameResult) -> None:
        expires_at = frame_result.frame_id + self.style.entity_display_frames
        for robot in frame_result.robots:
            if robot.position_metric is not None:
                self._visible_robots[robot.id] = (robot, expires_at)
        if frame_result.ball is not None and frame_result.ball.position_metric is not None:
            self._visible_ball = (frame_result.ball, expires_at)

    def _robots_for_frame(self, frame_id: int) -> List[Robot]:
        expired = [
            robot_id
            for robot_id, (_, expires_at) in self._visible_robots.items()
            if expires_at <= frame_id
        ]
        for robot_id in expired:
            self._visible_robots.pop(robot_id, None)
        return [robot for robot, _ in self._visible_robots.values()]

    def _ball_for_frame(self, frame_id: int) -> Optional[Ball]:
        if self._visible_ball is None:
            return None
        ball, expires_at = self._visible_ball
        if expires_at <= frame_id:
            self._visible_ball = None
            return None
        return ball

    def draw_field(self) -> "np.ndarray":
        w, h = self.style.output_size
        canvas = np.full((h, w, 3), self.style.field_bgr, dtype=np.uint8)

        g = self.style.geometry
        margin = self.style.margin_px

        available_w = max(1, w - 2 * margin)
        available_h = max(1, h - 2 * margin - self.style.header_h_px)
        scale = min(available_w / g.outer_width_cm, available_h / g.outer_height_cm)

        field_w_px = int(round(g.outer_width_cm * scale))
        field_h_px = int(round(g.outer_height_cm * scale))

        x0 = (w - field_w_px) // 2
        y0 = self.style.header_h_px + (h - self.style.header_h_px - field_h_px) // 2
        x1 = x0 + field_w_px
        y1 = y0 + field_h_px

        # Muro exterior negro
        cv2.rectangle(
            canvas,
            (x0, y0),
            (x1, y1),
            self.style.wall_bgr,
            self._px(self.style.outer_wall_thickness_px),
            cv2.LINE_AA,
        )

        # Geometría de líneas internas (blancas)
        inset_px = self._cm_to_px(g.boundary_inset_cm, scale)
        ix0, iy0 = x0 + inset_px, y0 + inset_px
        ix1, iy1 = x1 - inset_px, y1 - inset_px
        icx, icy = (ix0 + ix1) // 2, (iy0 + iy1) // 2

        # Rectángulo interior blanco 219 x 158
        cv2.rectangle(
            canvas,
            (ix0, iy0),
            (ix1, iy1),
            self.style.line_bgr,
            self._px(self.style.inner_line_thickness_px),
            cv2.LINE_AA,
        )

        # Línea central punteada
        self._draw_dashed_line(
            canvas,
            (icx, iy0),
            (icx, iy1),
            self.style.center_mark_bgr,
            self._px(self.style.center_line_thickness_px),
            dash_px=max(6, self._px(10)),
            gap_px=max(6, self._px(8)),
        )

        # Círculo central: diámetro 60 cm
        center_r_px = self._cm_to_px(g.center_circle_diameter_cm / 2.0, scale)
        cv2.circle(
            canvas,
            (icx, icy),
            center_r_px,
            self.style.center_mark_bgr,
            self._px(self.style.circle_thickness_px),
            cv2.LINE_AA,
        )

        # Punto central
        spot_r_px = max(2, self._cm_to_px(g.center_spot_radius_cm, scale))
        cv2.circle(canvas, (icx, icy), spot_r_px, self.style.center_mark_bgr, -1, cv2.LINE_AA)

        # Áreas redondeadas blancas (como en la imagen)
        box_depth_px = self._cm_to_px(g.penalty_box_depth_cm, scale)
        box_h_px = self._cm_to_px(g.penalty_box_height_cm, scale)
        box_r_px = max(4, self._cm_to_px(g.penalty_box_corner_radius_cm, scale))
        top_y = icy - box_h_px // 2
        bottom_y = top_y + box_h_px

        self._draw_rounded_rect_outline(
            canvas,
            ix0,
            top_y,
            ix0 + box_depth_px,
            bottom_y,
            box_r_px,
            self.style.line_bgr,
            self._px(self.style.inner_line_thickness_px),
        )
        self._draw_rounded_rect_outline(
            canvas,
            ix1 - box_depth_px,
            top_y,
            ix1,
            bottom_y,
            box_r_px,
            self.style.line_bgr,
            self._px(self.style.inner_line_thickness_px),
        )

        # Porterias
        goal_depth_px = self._cm_to_px(g.goal_depth_cm, scale)
        goal_h_px = self._cm_to_px(g.goal_height_cm, scale)
        goal_y0 = icy - goal_h_px // 2
        goal_y1 = goal_y0 + goal_h_px

        # Centradas en el espacio entre muro exterior y línea blanca interior
        left_goal_x0 = x0 + max(1, (inset_px - goal_depth_px) // 2)
        left_goal_x1 = left_goal_x0 + goal_depth_px

        right_goal_x1 = x1 - max(1, (inset_px - goal_depth_px) // 2)
        right_goal_x0 = right_goal_x1 - goal_depth_px

        left_goal_color = self.style.rival_goal_bgr if self.style.mirror_x else self.style.ally_goal_bgr
        right_goal_color = self.style.ally_goal_bgr if self.style.mirror_x else self.style.rival_goal_bgr

        cv2.rectangle(
            canvas,
            (left_goal_x0, goal_y0),
            (left_goal_x1, goal_y1),
            left_goal_color,
            self._px(self.style.goal_thickness_px),
            cv2.LINE_AA,
        )

        cv2.rectangle(
            canvas,
            (right_goal_x0, goal_y0),
            (right_goal_x1, goal_y1),
            right_goal_color,
            self._px(self.style.goal_thickness_px),
            cv2.LINE_AA,
        )
            
        # Guarda estado geométrico para el mapeo de coordenadas
        self._field_origin_px = (x0, y0)
        self._field_scale_px_cm = scale
        return canvas

    def _append_trail(self, robot: Robot) -> None:
        p = robot.position_metric
        self._trails[robot.id].append((p.x, p.y))

    def _draw_trails(self, canvas: "np.ndarray") -> None:
        for points in self._trails.values():
            if len(points) < 2:
                continue
            pixel_points = [self._to_canvas_point(x, y) for x, y in points]
            for start, end in zip(pixel_points, pixel_points[1:]):
                cv2.line(canvas, start, end, self.style.trail_bgr, self._px(self.style.trail_thickness_px), cv2.LINE_AA)

    def _draw_robot(self, canvas: "np.ndarray", robot: Robot) -> None:
        p = robot.position_metric
        point = self._to_canvas_point(p.x, p.y)
        color = self._team_color(robot.team_id)
        radius = self._px(self.style.robot_radius_px)
        thickness = self._px(self.style.robot_circle_thickness_px) if not robot.is_penalized else -1
        cv2.circle(canvas, point, radius, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, point, radius, self.style.text_bgr, self._px(self.style.robot_outline_thickness_px), cv2.LINE_AA)
        angle = getattr(robot, "angle", 0.0) or 0.0
        if self.style.mirror_x:
            angle = math.pi - angle
        direction = (
            int(round(point[0] + np.cos(angle) * self._px(self.style.robot_direction_len_px))),
            int(round(point[1] + np.sin(angle) * self._px(self.style.robot_direction_len_px))),
        )
        cv2.arrowedLine(canvas, point, direction, self.style.text_bgr, 1, cv2.LINE_AA, tipLength=self.style.arrow_tip_length)
        ox, oy = self.style.robot_label_offset_px
        cv2.putText(
            canvas,
            robot.id,
            (point[0] + self._px(ox), point[1] + self._px(oy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self._ui_scale(self.style.robot_label_scale),
            self.style.text_bgr,
            1,
            cv2.LINE_AA,
        )

    def _draw_ball(self, canvas: "np.ndarray", ball: Ball) -> None:
        p = ball.position_metric
        point = self._to_canvas_point(p.x, p.y)
        radius = self._px(self.style.ball_radius_px)
        cv2.circle(canvas, point, radius, self.style.ball_bgr, -1, cv2.LINE_AA)
        cv2.circle(canvas, point, radius, self.style.text_bgr, self._px(self.style.ball_outline_thickness_px), cv2.LINE_AA)
        dx, dy = getattr(ball, "direction_vector", (0.0, 0.0))
        if (dx, dy) != (0.0, 0.0):
            if self.style.mirror_x:
                dx = -dx
            end = (
                int(round(point[0] + dx * self._px(self.style.ball_direction_len_px))),
                int(round(point[1] + dy * self._px(self.style.ball_direction_len_px))),
            )
            cv2.arrowedLine(canvas, point, end, self.style.ball_bgr, 1, cv2.LINE_AA, tipLength=self.style.arrow_tip_length)

    def _draw_events(
        self,
        canvas: "np.ndarray",
        frame_events: Optional[FrameEvents],
        frame_id: int,
        frame_result: FrameResult,
    ) -> None:
        if frame_events is not None:
            for event in frame_events.eventos[-self.style.event_max_visible:]:
                key = self._event_key(event)
                already_active = any(active_key == key for active_key, _, _ in self._active_events)
                if not already_active:
                    self._active_events.append(
                        (key, event, frame_id + self.style.event_display_frames)
                    )

        while self._active_events and self._active_events[0][2] <= frame_id:
            self._active_events.popleft()

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
                self._draw_pass_event(canvas, event, robot_by_id, color)
            else:
                position = self._event_position(event)
                if position is not None:
                    self._draw_event_marker(canvas, event_type, position, color)

            label = self._event_label(event)
            if label not in labels:
                labels.append(label)

        y = self.style.margin_px + self.style.header_h_px + self._px(14)
        for label in labels:
            cv2.putText(
                canvas,
                label,
                (self.style.margin_px, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self._ui_scale(self.style.event_label_scale),
                self.style.text_bgr,
                1,
                cv2.LINE_AA,
            )
            y += self._px(self.style.event_label_line_height_px)


    def _draw_pass_event(
        self,
        canvas: "np.ndarray",
        event: object,
        robot_by_id: Dict[str, Robot],
        color: Color,
    ) -> None:
        from_id = getattr(event, "from_robot_id", getattr(event, "from_id", None))
        to_id = getattr(event, "to_robot_id", getattr(event, "to", None))

        if from_id not in robot_by_id or to_id not in robot_by_id:
            return

        p1 = robot_by_id[from_id].position_metric
        p2 = robot_by_id[to_id].position_metric

        start = self._to_canvas_point(p1.x, p1.y)
        end = self._to_canvas_point(p2.x, p2.y)

        cv2.arrowedLine(
            canvas,
            start,
            end,
            color,
            self._px(2),
            cv2.LINE_AA,
            tipLength=self.style.arrow_tip_length,
        )


    def _draw_event_marker(
        self,
        canvas: "np.ndarray",
        event_type: str,
        position_cm: Tuple[float, float],
        color: Color,
    ) -> None:
        point = self._to_canvas_point(position_cm[0], position_cm[1])
        radius = self._px(self.style.event_radius_px)

        if event_type == "gol_valido":
            cv2.drawMarker(
                canvas,
                point,
                color,
                markerType=cv2.MARKER_STAR,
                markerSize=radius * 2,
                thickness=self._px(2),
                line_type=cv2.LINE_AA,
            )
            return

        if event_type == "gol_invalido":
            cv2.drawMarker(
                canvas,
                point,
                color,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=radius * 2,
                thickness=self._px(2),
                line_type=cv2.LINE_AA,
            )
            return

        if event_type == "colision":
            cv2.circle(canvas, point, radius, color, self._px(2), cv2.LINE_AA)
            cv2.drawMarker(
                canvas,
                point,
                color,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=radius,
                thickness=self._px(2),
                line_type=cv2.LINE_AA,
            )
            return

        cv2.circle(canvas, point, radius, color, self._px(2), cv2.LINE_AA)


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

    def _to_canvas_point(self, x_cm: float, y_cm: float) -> Tuple[int, int]:
        x0, y0 = getattr(self, "_field_origin_px", (self.style.margin_px, self.style.margin_px + self.style.header_h_px))
        s = getattr(self, "_field_scale_px_cm", 1.0)
        if self.style.mirror_x:
            x_cm = self.style.geometry.outer_width_cm - x_cm
        return (int(round(x0 + x_cm * s)), int(round(y0 + y_cm * s)))

    def _team_color(self, team_id: str) -> Color:
        normalized = team_id.lower()
        if normalized in {"ally", "allies", "azul", "blue"}:
            return self.style.ally_bgr
        if normalized in {"rival", "rivals", "rojo", "red"}:
            return self.style.rival_bgr
        return self.style.unknown_bgr

    def _cm_to_px(self, value_cm: float, scale: float) -> int:
        return max(1, int(round(value_cm * scale)))

    def _px(self, value_px: int) -> int:
        ref = max(0.7, min(1.8, self.style.output_size[1] / float(self.style.base_ref_h)))
        return max(1, int(round(value_px * ref)))

    def _ui_scale(self, scale: float) -> float:
        ref = max(0.75, min(1.45, self.style.output_size[1] / float(self.style.base_ref_h)))
        return scale * ref

    def _draw_dashed_line(
        self,
        canvas: "np.ndarray",
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: Color,
        thickness: int,
        dash_px: int,
        gap_px: int,
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        length = int(np.hypot(x2 - x1, y2 - y1))
        if length <= 0:
            return
        vx = (x2 - x1) / length
        vy = (y2 - y1) / length
        pos = 0
        while pos < length:
            p0 = pos
            p1 = min(pos + dash_px, length)
            sx = int(round(x1 + vx * p0))
            sy = int(round(y1 + vy * p0))
            ex = int(round(x1 + vx * p1))
            ey = int(round(y1 + vy * p1))
            cv2.line(canvas, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
            pos += dash_px + gap_px

    def _draw_rounded_rect_outline(
        self,
        canvas: "np.ndarray",
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        r: int,
        color: Color,
        thickness: int,
    ) -> None:
        r = min(r, max(1, (x1 - x0) // 2 - 1), max(1, (y1 - y0) // 2 - 1))

        cv2.line(canvas, (x0 + r, y0), (x1 - r, y0), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x0 + r, y1), (x1 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x0, y0 + r), (x0, y1 - r), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1, y0 + r), (x1, y1 - r), color, thickness, cv2.LINE_AA)

        cv2.ellipse(canvas, (x0 + r, y0 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x1 - r, y0 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x1 - r, y1 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (x0 + r, y1 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)


def draw_tactical_map(
    frame_result: FrameResult,
    frame_events: Optional[FrameEvents] = None,
    *,
    style: FieldStyle = FieldStyle(),
) -> "np.ndarray":
    return TacticalMapRenderer(style=style).render(frame_result, frame_events)
