from __future__ import annotations

from dataclasses import fields, is_dataclass
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Iterable

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
CODE_DIR = Path(__file__).resolve().parents[1]

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


# ---------------------------------------------------------------------------
# Imports del proyecto
# ---------------------------------------------------------------------------

from domain.events import (
    BallRepositionEvent,
    CollisionEvent,
    FrameEvents,
    GoalEvent,
    InvalidGoalEvent,
    OffCourtEvent,
    PanicEvent,
    PassEvent,
    PossessionEvent,
    RemoveRobotEvent,
    RobotRepositionEvent,
    RobotStoppedEvent,
)

from visualization.dashboard import render_dashboard
from visualization.layout import compose_final_frame
from visualization.tactical_map import FieldStyle, TacticalMapRenderer
from visualization.video_render import render_video_overlay


# ---------------------------------------------------------------------------
# Paths del proyecto
# ---------------------------------------------------------------------------

DATA_DIR = CODE_DIR / "data"
OUTPUTS_DIR = DATA_DIR / "outputs"

MOCK_OUTPUT_DIR = OUTPUTS_DIR / "mock_visualization"

FRAME_006_PATH = MOCK_OUTPUT_DIR / "frame_006.png"
LAST_FRAME_PATH = MOCK_OUTPUT_DIR / "frame_last.png"
VIDEO_PATH = MOCK_OUTPUT_DIR / "mock_visualization.mp4"


# ---------------------------------------------------------------------------
# Configuración mock
# ---------------------------------------------------------------------------

FPS = 30
TOTAL_FRAMES = 90

FIELD_WIDTH_CM = 243.0
FIELD_HEIGHT_CM = 182.0

OUTPUT_SIZE = (1600, 900)  # width, height
LEFT_WIDTH_RATIO = 0.64
VIDEO_PREVIEW_HEIGHT_RATIO = 0.50



# ---------------------------------------------------------------------------
# Helpers genéricos
# ---------------------------------------------------------------------------

def _point(x: float, y: float) -> SimpleNamespace:
    return SimpleNamespace(x=float(x), y=float(y))


def _build(cls: type, **kwargs: Any) -> Any:
    """Construye dataclasses filtrando campos inválidos.

    Sirve para que el mock no truene si algún evento todavía conserva nombres
    viejos como `from_id`, `to`, `position`, etc.
    """
    if is_dataclass(cls):
        allowed = {field.name for field in fields(cls) if field.init}
        payload = {key: value for key, value in kwargs.items() if key in allowed}
        return cls(**payload)

    return cls(**kwargs)


def _attach_compat_attrs(obj: Any, **attrs: Any) -> Any:
    """Agrega atributos legacy si el objeto lo permite.

    Esto ayuda si `tactical_map.py` todavía busca `position`, `last_position`
    o `target_position`, mientras tus eventos nuevos ya usan `position_cm`,
    `last_position_cm` o `target_position_cm`.
    """
    for key, value in attrs.items():
        try:
            current = getattr(obj, key, None)
            if current is None:
                setattr(obj, key, value)
        except Exception:
            pass

    return obj


def _metric_to_pixel(x_cm: float, y_cm: float, video_size: tuple[int, int]) -> SimpleNamespace:
    width, height = video_size
    x_px = x_cm * width / FIELD_WIDTH_CM
    y_px = y_cm * height / FIELD_HEIGHT_CM
    return _point(x_px, y_px)


def _make_stats(frame_id: int) -> SimpleNamespace:
    """Objeto stats compatible con dashboard.py.

    No uso domain.stats directamente para evitar depender de constructores.
    dashboard.py solo necesita atributos:
    - score.allies
    - score.rivals
    - possession_pct.allies
    - possession_pct.rivals
    - distance_cm
    """
    allies_score = 2 if frame_id >= 55 else 1 if frame_id >= 23 else 0
    rivals_score = 1 if frame_id >= 13 else 0

    return SimpleNamespace(
        score=SimpleNamespace(
            allies=allies_score,
            rivals=rivals_score,
        ),
        possession_pct=SimpleNamespace(
            allies=58.0,
            rivals=42.0,
        ),
        distance_cm={
            "A1": 520.0 + frame_id * 2.3,
            "A2": 430.0 + frame_id * 1.8,
            "R1": 390.0 + frame_id * 2.0,
            "R2": 365.0 + frame_id * 1.5,
            "allies": 1340.0 + frame_id * 4.0,
            "rivals": 980.0 + frame_id * 3.2,
        },
    )


# ---------------------------------------------------------------------------
# Mock de entidades
# ---------------------------------------------------------------------------

def _make_robot(
    robot_id: str,
    team_id: str,
    x_cm: float,
    y_cm: float,
    angle: float,
    video_size: tuple[int, int],
    *,
    is_penalized: bool = False,
) -> SimpleNamespace:
    position_metric = _point(x_cm, y_cm)
    position_pixel = _metric_to_pixel(x_cm, y_cm, video_size)

    return SimpleNamespace(
        id=robot_id,
        team_id=team_id,
        position_metric=position_metric,
        position_pixel=position_pixel,
        angle=angle,
        is_penalized=is_penalized,
    )


def _make_ball(
    x_cm: float,
    y_cm: float,
    video_size: tuple[int, int],
    *,
    direction_vector: tuple[float, float] = (1.0, 0.0),
) -> SimpleNamespace:
    return SimpleNamespace(
        position_metric=_point(x_cm, y_cm),
        position_pixel=_metric_to_pixel(x_cm, y_cm, video_size),
        direction_vector=direction_vector,
    )


def _mock_robots(frame_id: int, video_size: tuple[int, int]) -> list[SimpleNamespace]:
    t = frame_id / FPS

    # Movimiento suave para que los trails del mapa se vean.
    a1_x = 62.0 + 18.0 * math.sin(t * 1.3)
    a1_y = 72.0 + 10.0 * math.cos(t * 1.1)

    a2_x = 162.0 + 13.0 * math.sin(t * 1.0 + 1.2)
    a2_y = 73.0 + 11.0 * math.cos(t * 1.4)

    r1_x = 119.0 + 10.0 * math.sin(t * 0.8 + 2.1)
    r1_y = 129.0 + 13.0 * math.cos(t * 1.2)

    r2_x = 210.0 + 12.0 * math.sin(t * 1.5)
    r2_y = 109.0 + 10.0 * math.cos(t * 1.0 + 0.8)

    return [
        _make_robot("A1", "allies", a1_x, a1_y, 0.55, video_size),
        _make_robot("A2", "allies", a2_x, a2_y, 1.10, video_size),
        _make_robot("R1", "rivals", r1_x, r1_y, -2.55, video_size),
        _make_robot(
            "R2",
            "rivals",
            r2_x,
            r2_y,
            2.85,
            video_size,
            is_penalized=frame_id >= 52,
        ),
    ]


def _mock_ball(frame_id: int, video_size: tuple[int, int]) -> SimpleNamespace:
    t = frame_id / FPS

    x = 98.0 + 42.0 * math.sin(t * 1.5)
    y = 91.0 + 17.0 * math.cos(t * 1.2)

    return _make_ball(
        x,
        y,
        video_size,
        direction_vector=(0.9, 0.25),
    )


def _mock_frame_result(frame_id: int, video_size: tuple[int, int]) -> SimpleNamespace:
    robots = _mock_robots(frame_id, video_size)
    ball = _mock_ball(frame_id, video_size)

    return SimpleNamespace(
        frame_id=frame_id,
        robots=robots,
        ball=ball,
    )


# ---------------------------------------------------------------------------
# Mock de eventos
# ---------------------------------------------------------------------------

def _robot_by_id(robots: Iterable[Any]) -> dict[str, Any]:
    return {robot.id: robot for robot in robots}


def _event_time(frame: int) -> float:
    return frame / FPS


def _mock_events(frame_id: int, robots: list[Any]) -> FrameEvents:
    robot_by_id = _robot_by_id(robots)
    eventos: list[Any] = []

    def add_at(event_frame: int, event: Any) -> None:
        if frame_id >= event_frame:
            eventos.append(event)

    # F3 — posesión
    pos_possession = (72.0, 83.0)
    possession = _build(
        PossessionEvent,
        frame=3,
        timestamp_s=_event_time(3),
        position_cm=pos_possession,
        position=pos_possession,
        robot=robot_by_id["A1"],
        team="allies",
        distance_cm=8.5,
        severity="info",
        confidence=0.94,
    )
    _attach_compat_attrs(possession, position=pos_possession)
    add_at(3, possession)

    # F6 — pase
    pass_pos = (116.0, 91.0)
    pass_event = _build(
        PassEvent,
        frame=6,
        timestamp_s=_event_time(6),
        position_cm=pass_pos,
        position=pass_pos,
        from_robot_id="A1",
        to_robot_id="A2",
        from_id="A1",
        to="A2",
        from_team="allies",
        to_team="allies",
        distance_cm=64.0,
        successful=True,
        severity="info",
        confidence=0.89,
    )
    _attach_compat_attrs(
        pass_event,
        position=pass_pos,
        from_id="A1",
        to="A2",
    )
    add_at(6, pass_event)

    # F9 — colisión
    collision_pos = (120.0, 132.0)
    collision = _build(
        CollisionEvent,
        frame=9,
        timestamp_s=_event_time(9),
        position_cm=collision_pos,
        position=collision_pos,
        robots=[robot_by_id["R1"], robot_by_id["A1"]],
        impact_speed_cm_s=32.0,
        severity="warning",
        confidence=0.91,
    )
    _attach_compat_attrs(collision, position=collision_pos)
    add_at(9, collision)

    # F13 — gol inválido
    invalid_goal_pos = (225.0, 91.0)
    invalid_goal = _build(
        InvalidGoalEvent,
        frame=13,
        timestamp_s=_event_time(13),
        position_cm=invalid_goal_pos,
        position=invalid_goal_pos,
        team="rivals",
        reason="robot_en_area_penalti",
        infracting_robot=robot_by_id["R2"],
        severity="danger",
        confidence=0.86,
    )
    _attach_compat_attrs(invalid_goal, position=invalid_goal_pos)
    add_at(13, invalid_goal)

    # F17 — reposición de balón
    ball_reposition_from = (121.5, 91.0)
    ball_reposition_to = (121.5, 91.0)
    ball_reposition = _build(
        BallRepositionEvent,
        frame=17,
        timestamp_s=_event_time(17),
        position_cm=ball_reposition_to,
        position=ball_reposition_to,
        reason="falta_de_progreso",
        from_position_cm=ball_reposition_from,
        to_position_cm=ball_reposition_to,
        severity="warning",
        confidence=0.95,
    )
    _attach_compat_attrs(ball_reposition, position=ball_reposition_to)
    add_at(17, ball_reposition)

    # F23 — gol válido
    goal_pos = (231.0, 91.0)
    goal = _build(
        GoalEvent,
        frame=23,
        timestamp_s=_event_time(23),
        position_cm=goal_pos,
        position=goal_pos,
        team="allies",
        velocity_cm_s=95.0,
        scorer_robot=robot_by_id["A2"],
        severity="success",
        confidence=0.97,
    )
    _attach_compat_attrs(goal, position=goal_pos)
    add_at(23, goal)

    # F31 — robot detenido
    stopped_pos = (
        robot_by_id["R1"].position_metric.x,
        robot_by_id["R1"].position_metric.y,
    )
    stopped = _build(
        RobotStoppedEvent,
        frame=31,
        timestamp_s=_event_time(31),
        position_cm=stopped_pos,
        position=stopped_pos,
        robot=robot_by_id["R1"],
        frames_duration=45,
        duration_s=45 / FPS,
        severity="warning",
        confidence=0.88,
    )
    _attach_compat_attrs(stopped, position=stopped_pos)
    add_at(31, stopped)

    # F37 — balón fuera de cancha
    offcourt_pos = (244.5, 96.0)
    offcourt = _build(
        OffCourtEvent,
        frame=37,
        timestamp_s=_event_time(37),
        position_cm=offcourt_pos,
        position=offcourt_pos,
        object_type="balon",
        object="balon",
        object_id=None,
        last_position_cm=offcourt_pos,
        last_position=offcourt_pos,
        severity="warning",
        confidence=0.92,
    )
    _attach_compat_attrs(
        offcourt,
        position=offcourt_pos,
        last_position=offcourt_pos,
    )
    add_at(37, offcourt)

    # F45 — reposición de robot
    robot_reposition_from = (
        robot_by_id["R2"].position_metric.x,
        robot_by_id["R2"].position_metric.y,
    )
    robot_reposition_to = (210.0, 91.0)
    robot_reposition = _build(
        RobotRepositionEvent,
        frame=45,
        timestamp_s=_event_time(45),
        position_cm=robot_reposition_to,
        position=robot_reposition_to,
        robot=robot_by_id["R2"],
        from_position_cm=robot_reposition_from,
        target_position_cm=robot_reposition_to,
        target_position=robot_reposition_to,
        severity="warning",
        confidence=0.90,
    )
    _attach_compat_attrs(
        robot_reposition,
        position=robot_reposition_to,
        target_position=robot_reposition_to,
    )
    add_at(45, robot_reposition)

    # F52 — sacar robot
    remove_robot_pos = (
        robot_by_id["R2"].position_metric.x,
        robot_by_id["R2"].position_metric.y,
    )
    remove_robot = _build(
        RemoveRobotEvent,
        frame=52,
        timestamp_s=_event_time(52),
        position_cm=remove_robot_pos,
        position=remove_robot_pos,
        robot=robot_by_id["R2"],
        reason="toco_pared_cancha",
        frames_penalization=90,
        penalization_s=90 / FPS,
        severity="danger",
        confidence=0.93,
    )
    _attach_compat_attrs(remove_robot, position=remove_robot_pos)
    add_at(52, remove_robot)

    # F58 — panic
    panic_pos = (121.5, 91.0)
    panic = _build(
        PanicEvent,
        frame=58,
        timestamp_s=_event_time(58),
        position_cm=panic_pos,
        position=panic_pos,
        reason="ruido_visual",
        frames_duration=18,
        duration_s=18 / FPS,
        severity="system",
        confidence=0.75,
    )
    _attach_compat_attrs(panic, position=panic_pos)
    add_at(58, panic)

    return _build(
        FrameEvents,
        frame=frame_id,
        timestamp_s=_event_time(frame_id),
        eventos=eventos,
    )


# ---------------------------------------------------------------------------
# Mock video frame
# ---------------------------------------------------------------------------

def _mock_video_frame(_frame_id: int, video_size: tuple[int, int]) -> np.ndarray:
    width, height = video_size
    return np.full((height, width, 3), (28, 34, 42), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Render principal
# ---------------------------------------------------------------------------







def render_mock_frame(
    frame_id: int,
    tactical_renderer: TacticalMapRenderer,
) -> np.ndarray:
    output_w, output_h = OUTPUT_SIZE

    left_w = int(round(output_w * LEFT_WIDTH_RATIO))
    dashboard_w = output_w - left_w

    video_h = int(round(output_h * VIDEO_PREVIEW_HEIGHT_RATIO))
    tactical_h = output_h - video_h

    video_size = (left_w, video_h)

    frame_result = _mock_frame_result(frame_id, video_size)
    frame_events = _mock_events(frame_id, frame_result.robots)
    stats = _make_stats(frame_id)
    match_time_seconds = frame_id / FPS

    raw_video = _mock_video_frame(frame_id, video_size)
    video_overlay = render_video_overlay(
        raw_video,
        frame_result,
        frame_events,
    )

    tactical_map = tactical_renderer.render(frame_result, frame_events)

    dashboard = render_dashboard(
        stats,
        frame_events,
        match_time_seconds,
        dashboard_w,
        output_h,
    )

    return compose_final_frame(
        video_overlay,
        tactical_map,
        dashboard,
        output_size=OUTPUT_SIZE,
        left_width_ratio=LEFT_WIDTH_RATIO,
        video_preview_height_ratio=VIDEO_PREVIEW_HEIGHT_RATIO,
    )


def main() -> None:
    MOCK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_w, output_h = OUTPUT_SIZE
    left_w = int(round(output_w * LEFT_WIDTH_RATIO))
    video_h = int(round(output_h * VIDEO_PREVIEW_HEIGHT_RATIO))
    tactical_h = output_h - video_h

    tactical_style = FieldStyle(output_size=(left_w, tactical_h))
    tactical_renderer = TacticalMapRenderer(
        style=tactical_style,
        trail_length=60,
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(VIDEO_PATH),
        fourcc,
        FPS,
        OUTPUT_SIZE,
    )

    last_frame = None

    for frame_id in range(1, TOTAL_FRAMES + 1):
        final_frame = render_mock_frame(frame_id, tactical_renderer)

        if frame_id == 6:
            cv2.imwrite(str(FRAME_006_PATH), final_frame)

        writer.write(final_frame)
        last_frame = final_frame

    writer.release()

    if last_frame is not None:
        cv2.imwrite(str(LAST_FRAME_PATH), last_frame)

    print(f"OK frame 006: {FRAME_006_PATH}")
    print(f"OK last frame: {LAST_FRAME_PATH}")
    print(f"OK video: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
