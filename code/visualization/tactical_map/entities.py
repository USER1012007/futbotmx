from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Tuple

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("visualization.tactical_map.entities requires opencv-python.") from exc

from visualization.common.drawing import team_color
from visualization.tactical_map.field import FieldRenderContext, scale_px, to_canvas_point, ui_scale
from visualization.tactical_map.style import FieldStyle

if TYPE_CHECKING:
    from domain.entities import Ball, Robot


TrailPoints = Iterable[Tuple[float, float]]


def draw_trails(
    canvas: "object",
    style: FieldStyle,
    context: FieldRenderContext,
    trails: Iterable[TrailPoints],
) -> None:
    for points_iter in trails:
        points = list(points_iter)
        if len(points) < 2:
            continue
        pixel_points = [to_canvas_point(style, context, x, y) for x, y in points]
        for start, end in zip(pixel_points, pixel_points[1:]):
            cv2.line(canvas, start, end, style.trail_bgr, scale_px(style, style.trail_thickness_px), cv2.LINE_AA)


def draw_robot(canvas: "object", style: FieldStyle, context: FieldRenderContext, robot: "Robot") -> None:
    point = to_canvas_point(style, context, robot.position_metric.x, robot.position_metric.y)
    color = team_color(robot.team_id, style.ally_bgr, style.rival_bgr, style.unknown_bgr)
    radius = scale_px(style, style.robot_radius_px)
    thickness = scale_px(style, style.robot_circle_thickness_px) if not robot.is_penalized else -1
    cv2.circle(canvas, point, radius, color, thickness, cv2.LINE_AA)
    cv2.circle(canvas, point, radius, style.text_bgr, scale_px(style, style.robot_outline_thickness_px), cv2.LINE_AA)

    ox, oy = style.robot_label_offset_px
    cv2.putText(
        canvas,
        robot.id,
        (point[0] + scale_px(style, ox), point[1] + scale_px(style, oy)),
        cv2.FONT_HERSHEY_SIMPLEX,
        ui_scale(style, style.robot_label_scale),
        style.text_bgr,
        1,
        cv2.LINE_AA,
    )


def draw_ball(canvas: "object", style: FieldStyle, context: FieldRenderContext, ball: "Ball") -> None:
    point = to_canvas_point(style, context, ball.position_metric.x, ball.position_metric.y)
    radius = scale_px(style, style.ball_radius_px)
    cv2.circle(canvas, point, radius, style.ball_bgr, -1, cv2.LINE_AA)
    cv2.circle(canvas, point, radius, style.text_bgr, scale_px(style, style.ball_outline_thickness_px), cv2.LINE_AA)

    dx, dy = getattr(ball, "direction_vector", (0.0, 0.0))
    if (dx, dy) == (0.0, 0.0):
        return
    if style.mirror_x:
        dx = -dx
    end = (
        int(round(point[0] + dx * scale_px(style, style.ball_direction_len_px))),
        int(round(point[1] + dy * scale_px(style, style.ball_direction_len_px))),
    )
    cv2.arrowedLine(canvas, point, end, style.ball_bgr, 1, cv2.LINE_AA, tipLength=style.arrow_tip_length)


def draw_frame_label(canvas: "object", style: FieldStyle, frame_id: int) -> None:
    cv2.putText(
        canvas,
        f"Frame {frame_id}",
        (max(12, style.margin_px), max(20, style.margin_px + 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        ui_scale(style, style.header_scale),
        style.text_bgr,
        1,
        cv2.LINE_AA,
    )
