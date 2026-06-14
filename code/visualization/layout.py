from __future__ import annotations

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on runtime environment.
    raise ImportError("visualization.layout requires opencv-python and numpy.") from exc


def compose_final_frame(
    video_overlay: "np.ndarray",
    tactical_map: "np.ndarray",
    dashboard: "np.ndarray",
) -> "np.ndarray":
    """Compose video overlay, tactical map, and dashboard into one BGR frame."""
    if video_overlay.ndim != 3 or tactical_map.ndim != 3 or dashboard.ndim != 3:
        raise ValueError("All inputs must be HxWxC images.")

    left_w = video_overlay.shape[1]
    tactical_resized = cv2.resize(
        tactical_map,
        (left_w, _height_for_width(tactical_map, left_w)),
        interpolation=cv2.INTER_AREA,
    )

    left_col = np.vstack([video_overlay, tactical_resized])
    total_h = left_col.shape[0]

    # left_col debe ocupar 60% del ancho total -> dashboard ocupa 40%
    total_w = max(1, int(round(left_w / 0.6)))
    dashboard_w = total_w - left_w

    dashboard_resized = cv2.resize(
        dashboard,
        (dashboard_w, total_h),
        interpolation=cv2.INTER_AREA,
    )

    return np.hstack([left_col, dashboard_resized])


def _width_for_height(image: "np.ndarray", target_h: int) -> int:
    return max(1, int(round(image.shape[1] * (target_h / image.shape[0]))))


def _height_for_width(image: "np.ndarray", target_w: int) -> int:
    return max(1, int(round(image.shape[0] * (target_w / image.shape[1]))))
