from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise ImportError("visualization.tactical_map.field requires opencv-python and numpy.") from exc

from visualization.common.types import Color
from visualization.tactical_map.style import FieldStyle


@dataclass(frozen=True)
class FieldRenderContext:
    origin_px: Tuple[int, int]
    scale_px_cm: float


def draw_field(style: FieldStyle) -> tuple["np.ndarray", FieldRenderContext]:
    width, height = style.output_size
    canvas = np.full((height, width, 3), style.field_bgr, dtype=np.uint8)
    metrics = _field_metrics(style)

    _draw_outer_wall(canvas, style, metrics)
    _draw_inner_lines(canvas, style, metrics)
    _draw_center_marks(canvas, style, metrics)
    _draw_penalty_boxes(canvas, style, metrics)
    _draw_goals(canvas, style, metrics)

    return canvas, FieldRenderContext(
        origin_px=(int(metrics["x0"]), int(metrics["y0"])),
        scale_px_cm=float(metrics["scale"]),
    )


def to_canvas_point(style: FieldStyle, context: FieldRenderContext, x_cm: float, y_cm: float) -> Tuple[int, int]:
    x0, y0 = context.origin_px
    if style.mirror_x:
        x_cm = style.geometry.outer_width_cm - x_cm
    return (
        int(round(x0 + x_cm * context.scale_px_cm)),
        int(round(y0 + y_cm * context.scale_px_cm)),
    )


def scale_px(style: FieldStyle, value_px: int) -> int:
    ref = max(0.7, min(1.8, style.output_size[1] / float(style.base_ref_h)))
    return max(1, int(round(value_px * ref)))


def ui_scale(style: FieldStyle, scale: float) -> float:
    ref = max(0.75, min(1.45, style.output_size[1] / float(style.base_ref_h)))
    return scale * ref


def cm_to_px(value_cm: float, scale: float) -> int:
    return max(1, int(round(value_cm * scale)))


def _field_metrics(style: FieldStyle) -> Dict[str, int | float]:
    width, height = style.output_size
    geometry = style.geometry
    margin = style.margin_px

    available_w = max(1, width - 2 * margin)
    available_h = max(1, height - 2 * margin - style.header_h_px)
    scale = min(available_w / geometry.outer_width_cm, available_h / geometry.outer_height_cm)

    field_w_px = int(round(geometry.outer_width_cm * scale))
    field_h_px = int(round(geometry.outer_height_cm * scale))
    x0 = (width - field_w_px) // 2
    y0 = style.header_h_px + (height - style.header_h_px - field_h_px) // 2
    x1 = x0 + field_w_px
    y1 = y0 + field_h_px

    inset_px = cm_to_px(geometry.boundary_inset_cm, scale)
    ix0, iy0 = x0 + inset_px, y0 + inset_px
    ix1, iy1 = x1 - inset_px, y1 - inset_px
    icx, icy = (ix0 + ix1) // 2, (iy0 + iy1) // 2

    return {
        "scale": scale,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "inset_px": inset_px,
        "ix0": ix0,
        "iy0": iy0,
        "ix1": ix1,
        "iy1": iy1,
        "icx": icx,
        "icy": icy,
    }


def _draw_outer_wall(canvas: "np.ndarray", style: FieldStyle, metrics: Dict[str, int | float]) -> None:
    cv2.rectangle(
        canvas,
        (int(metrics["x0"]), int(metrics["y0"])),
        (int(metrics["x1"]), int(metrics["y1"])),
        style.wall_bgr,
        scale_px(style, style.outer_wall_thickness_px),
        cv2.LINE_AA,
    )


def _draw_inner_lines(canvas: "np.ndarray", style: FieldStyle, metrics: Dict[str, int | float]) -> None:
    cv2.rectangle(
        canvas,
        (int(metrics["ix0"]), int(metrics["iy0"])),
        (int(metrics["ix1"]), int(metrics["iy1"])),
        style.line_bgr,
        scale_px(style, style.inner_line_thickness_px),
        cv2.LINE_AA,
    )
    _draw_dashed_line(
        canvas,
        (int(metrics["icx"]), int(metrics["iy0"])),
        (int(metrics["icx"]), int(metrics["iy1"])),
        style.center_mark_bgr,
        scale_px(style, style.center_line_thickness_px),
        dash_px=max(6, scale_px(style, 10)),
        gap_px=max(6, scale_px(style, 8)),
    )


def _draw_center_marks(canvas: "np.ndarray", style: FieldStyle, metrics: Dict[str, int | float]) -> None:
    geometry = style.geometry
    scale = float(metrics["scale"])
    center = (int(metrics["icx"]), int(metrics["icy"]))
    center_r_px = cm_to_px(geometry.center_circle_diameter_cm / 2.0, scale)
    cv2.circle(canvas, center, center_r_px, style.center_mark_bgr, scale_px(style, style.circle_thickness_px), cv2.LINE_AA)
    spot_r_px = max(2, cm_to_px(geometry.center_spot_radius_cm, scale))
    cv2.circle(canvas, center, spot_r_px, style.center_mark_bgr, -1, cv2.LINE_AA)


def _draw_penalty_boxes(canvas: "np.ndarray", style: FieldStyle, metrics: Dict[str, int | float]) -> None:
    geometry = style.geometry
    scale = float(metrics["scale"])
    box_depth_px = cm_to_px(geometry.penalty_box_depth_cm, scale)
    box_h_px = cm_to_px(geometry.penalty_box_height_cm, scale)
    box_r_px = max(4, cm_to_px(geometry.penalty_box_corner_radius_cm, scale))
    top_y = int(metrics["icy"]) - box_h_px // 2
    bottom_y = top_y + box_h_px

    _draw_rounded_rect_outline(
        canvas,
        int(metrics["ix0"]),
        top_y,
        int(metrics["ix0"]) + box_depth_px,
        bottom_y,
        box_r_px,
        style.line_bgr,
        scale_px(style, style.inner_line_thickness_px),
    )
    _draw_rounded_rect_outline(
        canvas,
        int(metrics["ix1"]) - box_depth_px,
        top_y,
        int(metrics["ix1"]),
        bottom_y,
        box_r_px,
        style.line_bgr,
        scale_px(style, style.inner_line_thickness_px),
    )


def _draw_goals(canvas: "np.ndarray", style: FieldStyle, metrics: Dict[str, int | float]) -> None:
    geometry = style.geometry
    scale = float(metrics["scale"])
    goal_depth_px = cm_to_px(geometry.goal_depth_cm, scale)
    goal_h_px = cm_to_px(geometry.goal_height_cm, scale)
    goal_y0 = int(metrics["icy"]) - goal_h_px // 2
    goal_y1 = goal_y0 + goal_h_px
    inset_px = int(metrics["inset_px"])

    left_goal_x0 = int(metrics["x0"]) + max(1, (inset_px - goal_depth_px) // 2)
    left_goal_x1 = left_goal_x0 + goal_depth_px
    right_goal_x1 = int(metrics["x1"]) - max(1, (inset_px - goal_depth_px) // 2)
    right_goal_x0 = right_goal_x1 - goal_depth_px

    left_color = style.rival_goal_bgr if style.mirror_x else style.ally_goal_bgr
    right_color = style.ally_goal_bgr if style.mirror_x else style.rival_goal_bgr
    thickness = scale_px(style, style.goal_thickness_px)

    cv2.rectangle(canvas, (left_goal_x0, goal_y0), (left_goal_x1, goal_y1), left_color, thickness, cv2.LINE_AA)
    cv2.rectangle(canvas, (right_goal_x0, goal_y0), (right_goal_x1, goal_y1), right_color, thickness, cv2.LINE_AA)


def _draw_dashed_line(
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
    canvas: "np.ndarray",
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    radius: int,
    color: Color,
    thickness: int,
) -> None:
    radius = min(radius, max(1, (x1 - x0) // 2 - 1), max(1, (y1 - y0) // 2 - 1))
    cv2.line(canvas, (x0 + radius, y0), (x1 - radius, y0), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x0 + radius, y1), (x1 - radius, y1), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x0, y0 + radius), (x0, y1 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x1, y0 + radius), (x1, y1 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x0 + radius, y0 + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x1 - radius, y0 + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x1 - radius, y1 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x0 + radius, y1 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)
