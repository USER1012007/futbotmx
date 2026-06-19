from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from domain.field import FIELD_GEOMETRY, FieldGeometry as DomainFieldGeometry
from visualization.common.types import Color


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
class FieldStyle:
    geometry: DomainFieldGeometry = FIELD_GEOMETRY
    mirror_x: bool = False

    output_size: Tuple[int, int] = (1280, 720)
    margin_px: int = 12

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

    base_ref_h: int = 720
    outer_wall_thickness_px: int = 8
    inner_line_thickness_px: int = 5
    center_line_thickness_px: int = 3
    circle_thickness_px: int = 5
    goal_thickness_px: int = 4

    robot_radius_px: int = 18
    robot_direction_len_px: int = 32
    robot_label_scale: float = 0.8
    robot_label_offset_px: Tuple[int, int] = (18, 7)
    robot_outline_thickness_px: int = 1
    robot_circle_thickness_px: int = 2

    ball_radius_px: int = 10
    ball_direction_len_px: int = 26
    ball_outline_thickness_px: int = 1

    trail_thickness_px: int = 2

    event_radius_px: int = 24
    event_label_scale: float = 0.8
    event_label_line_height_px: int = 26
    event_display_frames: int = 75
    event_max_visible: int = 4
    entity_display_frames: int = 12
    header_scale: float = 0.8
    header_h_px: int = 24
    arrow_tip_length: float = 0.30
