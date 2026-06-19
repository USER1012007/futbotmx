from __future__ import annotations

from typing import Tuple

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("visualization.common.drawing requires opencv-python.") from exc

from visualization.common.types import Color


FONT = cv2.FONT_HERSHEY_SIMPLEX


def text(
    canvas: "object",
    value: str,
    origin: Tuple[int, int],
    scale: float,
    color: Color,
    thickness: int,
) -> None:
    cv2.putText(canvas, value, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def fit_text_scale(
    value: str,
    max_px: int,
    scale: float,
    thickness: int,
    *,
    min_scale: float = 0.35,
) -> float:
    if not value or max_px <= 0:
        return scale

    current = scale
    while current > min_scale:
        text_w, _ = cv2.getTextSize(value, FONT, current, thickness)[0]
        if text_w <= max_px:
            return current
        current *= 0.92

    return min_scale


def truncate_text(value: str, max_px: int, scale: float) -> str:
    if not value:
        return value

    text_w, _ = cv2.getTextSize(value, FONT, scale, 1)[0]
    if text_w <= max_px:
        return value

    suffix = "..."
    lo_i, hi_i = 0, len(value)

    while lo_i < hi_i:
        mid = (lo_i + hi_i + 1) // 2
        probe = value[:mid] + suffix
        probe_w, _ = cv2.getTextSize(probe, FONT, scale, 1)[0]

        if probe_w <= max_px:
            lo_i = mid
        else:
            hi_i = mid - 1

    return value[:lo_i] + suffix


def panel(
    canvas: "object",
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    fill_color: Color,
    border_color: Color,
) -> None:
    overlay = canvas.copy()
    radius = max(6, min(14, min(width, height) // 10))
    rounded_rect(overlay, x, y, width, height, radius, fill_color, -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    rounded_rect(canvas, x, y, width, height, radius, border_color, 1)


def rounded_rect(
    canvas: "object",
    x: int,
    y: int,
    width: int,
    height: int,
    radius: int,
    color: Color,
    thickness: int,
) -> None:
    x2 = x + width
    y2 = y + height
    radius = max(1, min(radius, width // 2, height // 2))

    if thickness < 0:
        cv2.rectangle(canvas, (x + radius, y), (x2 - radius, y2), color, thickness)
        cv2.rectangle(canvas, (x, y + radius), (x2, y2 - radius), color, thickness)
        cv2.circle(canvas, (x + radius, y + radius), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, (x2 - radius, y + radius), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, (x2 - radius, y2 - radius), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, (x + radius, y2 - radius), radius, color, thickness, cv2.LINE_AA)
        return

    cv2.line(canvas, (x + radius, y), (x2 - radius, y), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x, y + radius), (x, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x2 - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)


def team_color(team_id: str, ally_bgr: Color, rival_bgr: Color, unknown_bgr: Color) -> Color:
    normalized = team_id.lower()
    if normalized in {"ally", "allies", "azul", "blue"}:
        return ally_bgr
    if normalized in {"rival", "rivals", "rojo", "red"}:
        return rival_bgr
    return unknown_bgr
