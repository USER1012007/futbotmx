from __future__ import annotations

try:
    import cv2 as cv
    import numpy as np
except ImportError as exc:
    raise ImportError("visualization.layout requires opencv-python and numpy.") from exc


LEFT_WIDTH_RATIO = 0.64
VIDEO_PREVIEW_HEIGHT_RATIO = 0.50
BACKGROUND_BGR = (18, 22, 28)


def compose_final_frame(
    video_overlay: "np.ndarray",
    tactical_map: "np.ndarray",
    dashboard: "np.ndarray",
    *,
    output_size: tuple[int, int] | None = None,
    left_width_ratio: float = LEFT_WIDTH_RATIO,
    video_preview_height_ratio: float = VIDEO_PREVIEW_HEIGHT_RATIO,
) -> "np.ndarray":
    """Compose video preview, tactical map, and dashboard into one BGR frame.

    Layout:
    - Left column: video preview arriba + tactical map abajo con la misma altura.
    - Right column: dashboard.
    """
    if video_overlay.ndim != 3 or tactical_map.ndim != 3 or dashboard.ndim != 3:
        raise ValueError("All inputs must be HxWxC images.")

    if output_size is None:
        total_h = max(dashboard.shape[0], video_overlay.shape[0] + tactical_map.shape[0])
        total_w = max(
            video_overlay.shape[1] + dashboard.shape[1],
            int(round(video_overlay.shape[1] / left_width_ratio)),
        )
    else:
        total_w, total_h = output_size

    left_w = max(1, int(round(total_w * left_width_ratio)))
    dashboard_w = max(1, total_w - left_w)

    video_h = max(1, int(round(total_h * video_preview_height_ratio)))
    tactical_h = max(1, total_h - video_h)

    video_final = _resize_letterbox(video_overlay, left_w, video_h, BACKGROUND_BGR)
    tactical_final = _resize_letterbox(tactical_map, left_w, tactical_h, BACKGROUND_BGR)
    dashboard_final = _resize_letterbox(dashboard, dashboard_w, total_h, BACKGROUND_BGR)

    left_col = np.vstack([video_final, tactical_final])
    return np.hstack([left_col, dashboard_final])


def _resize_letterbox(
    image: "np.ndarray",
    target_w: int,
    target_h: int,
    bg_color: tuple[int, int, int],
) -> "np.ndarray":
    src_h, src_w = image.shape[:2]
    if src_w <= 0 or src_h <= 0:
        return np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas
