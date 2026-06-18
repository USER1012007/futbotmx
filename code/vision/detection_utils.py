import numpy as np
import supervision as sv


def combine_detections(
    detections: list[sv.Detections],
    *,
    default_class_id: int,
) -> sv.Detections:
    non_empty = [det for det in detections if len(det) > 0]
    if not non_empty:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.concatenate([det.xyxy for det in non_empty]),
        confidence=np.concatenate([
            det.confidence if det.confidence is not None else np.zeros(len(det), dtype=float)
            for det in non_empty
        ]),
        class_id=np.concatenate([
            det.class_id if det.class_id is not None else np.full(len(det), default_class_id)
            for det in non_empty
        ]),
        tracker_id=np.concatenate([
            det.tracker_id if det.tracker_id is not None else np.full(len(det), -1, dtype=int)
            for det in non_empty
        ]),
    )


def roi_bounds(
    width: int,
    height: int,
    center: tuple[float, float],
    size: int,
) -> tuple[int, int, int, int]:
    roi_w = min(size, width)
    roi_h = min(size, height)
    cx, cy = center
    x1 = int(round(cx - roi_w / 2))
    y1 = int(round(cy - roi_h / 2))
    x1 = max(0, min(width - roi_w, x1))
    y1 = max(0, min(height - roi_h, y1))
    return x1, y1, x1 + roi_w, y1 + roi_h


def scale_roi_detections_to_frame(
    det: sv.Detections,
    *,
    offset: tuple[int, int],
    roi_size: tuple[int, int],
    infer_size: tuple[int, int],
    class_id: int,
) -> sv.Detections:
    if len(det) == 0:
        return det

    scale_x = roi_size[0] / max(1, infer_size[0])
    scale_y = roi_size[1] / max(1, infer_size[1])
    xyxy = det.xyxy.copy()
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * scale_x + offset[0]
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * scale_y + offset[1]

    return sv.Detections(
        xyxy=xyxy,
        confidence=det.confidence,
        class_id=np.full(len(det), class_id, dtype=int),
        tracker_id=det.tracker_id,
    )


def offset_roi_detections_to_frame(
    det: sv.Detections,
    *,
    offset: tuple[int, int],
    class_id: int,
) -> sv.Detections:
    if len(det) == 0:
        return det

    xyxy = det.xyxy.copy()
    xyxy[:, [0, 2]] += offset[0]
    xyxy[:, [1, 3]] += offset[1]
    return sv.Detections(
        xyxy=xyxy,
        confidence=det.confidence,
        class_id=np.full(len(det), class_id, dtype=int),
        tracker_id=det.tracker_id,
    )
