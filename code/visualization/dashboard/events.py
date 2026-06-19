from __future__ import annotations

from collections import Counter
from typing import Dict, Mapping, TYPE_CHECKING, Tuple

from domain.stats import PossessionPct, Score, Statistics
from visualization.common.types import Color

if TYPE_CHECKING:
    from domain.events import FrameEvents


EVENT_LABELS: Mapping[str, str] = {
    "gol_valido": "Goles",
    "pase": "Pases",
    "colision": "Colisiones",
    "posesion": "Posesion",
    "fuera_de_cancha": "Fuera",
    "robot_detenido": "Detenidos",
    "reposicion_balon": "Rep. balon",
    "reposicion_robot": "Rep. robot",
    "gol_invalido": "Gol invalido",
    "sacar_robot": "Sacados",
    "panic": "Panic",
}

EVENT_PRIORITY = [
    "gol_valido",
    "gol_invalido",
    "pase",
    "colision",
    "reposicion_balon",
    "reposicion_robot",
    "sacar_robot",
    "fuera_de_cancha",
    "robot_detenido",
    "panic",
]

EVENT_COLORS: Mapping[str, Color] = {
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


def count_events(events: "FrameEvents") -> Dict[str, int]:
    return dict(
        Counter(
            getattr(event, "type", event.__class__.__name__)
            for event in getattr(events, "eventos", [])
        )
    )


def get_event_counts(stats: Statistics, events: "FrameEvents") -> Mapping[str, int]:
    counts = getattr(stats, "event_counts", None)
    return counts or count_events(events)


def event_key(event: object) -> tuple:
    event_type = getattr(event, "type", event.__class__.__name__)
    frame = getattr(event, "frame", None)
    robot = getattr(event, "robot", None)
    robot_id = getattr(robot, "id", None)
    position = getattr(event, "position_cm", getattr(event, "position", None))
    return (event_type, frame, robot_id, str(position))


def get_score(stats: Statistics) -> Score:
    return getattr(stats, "score", Score())


def get_possession(stats: Statistics) -> PossessionPct:
    return getattr(stats, "possession_pct", PossessionPct())


def get_distance(stats: Statistics) -> Mapping[str, float]:
    return getattr(stats, "distance_cm", {}) or {}


def normalize_pct(allies: float, rivals: float) -> Tuple[float, float]:
    total = allies + rivals
    if total <= 0:
        return 50.0, 50.0

    scale = 100.0 / total
    return allies * scale, rivals * scale


def format_time(seconds: float) -> str:
    safe = max(0, int(round(seconds)))
    minutes, secs = divmod(safe, 60)
    return f"{minutes:02d}:{secs:02d}"


def event_to_short_text(event: object) -> str:
    event_type = getattr(event, "type", event.__class__.__name__)
    prefix = event_time_prefix(event)

    if event_type == "gol_valido":
        team = getattr(event, "team", "?")
        return f"{prefix}Gol valido {team}"
    if event_type == "gol_invalido":
        team = getattr(event, "team", "?")
        reason = getattr(event, "reason", "")
        return f"{prefix}Gol invalido {team}: {reason}"
    if event_type == "pase":
        from_id = getattr(event, "from_robot_id", getattr(event, "from_id", "?"))
        to_id = getattr(event, "to_robot_id", getattr(event, "to", "?"))
        return f"{prefix}Pase {from_id} -> {to_id}"
    if event_type == "colision":
        robots = getattr(event, "robots", [])
        ids = "/".join(getattr(robot, "id", str(robot)) for robot in robots)
        return f"{prefix}Colision {ids}"
    if event_type == "posesion":
        robot = getattr(event, "robot", None)
        rid = getattr(robot, "id", "?")
        return f"{prefix}{rid} tomo posesion"
    if event_type == "fuera_de_cancha":
        object_type = getattr(event, "object_type", getattr(event, "object", "objeto"))
        return f"{prefix}{object_type} fuera de cancha"
    if event_type == "robot_detenido":
        robot = getattr(event, "robot", None)
        rid = getattr(robot, "id", "?")
        return f"{prefix}Robot {rid} detenido"
    if event_type == "reposicion_balon":
        reason = getattr(event, "reason", "")
        return f"{prefix}Reposicion balon: {reason}"
    if event_type == "reposicion_robot":
        robot = getattr(event, "robot", None)
        rid = getattr(robot, "id", "?")
        return f"{prefix}Reposicion robot {rid}"
    if event_type == "sacar_robot":
        robot = getattr(event, "robot", None)
        rid = getattr(robot, "id", "?")
        reason = getattr(event, "reason", "")
        return f"{prefix}Robot {rid} retirado: {reason}"
    if event_type == "panic":
        reason = getattr(event, "reason", "")
        return f"{prefix}Panic: {reason}"

    return f"{prefix}{EVENT_LABELS.get(event_type, event_type)}"


def event_time_prefix(event: object) -> str:
    timestamp_s = getattr(event, "timestamp_s", None)
    if timestamp_s is not None:
        return f"{format_time(float(timestamp_s))} - "

    frame = getattr(event, "frame", None)
    return f"F{frame} - " if frame is not None else ""


def event_color(event: object, default: Color) -> Color:
    event_type = getattr(event, "type", event.__class__.__name__)
    return EVENT_COLORS.get(event_type, default)
