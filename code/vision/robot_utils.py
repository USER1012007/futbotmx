import numpy as np
import supervision as sv
import cv2

from vision import cv_utils
from vision.segmentation_config import (
    CLASS_ROBOT,
    ROBOT_FALLBACK_CONFIDENCE,
    ROBOT_FALLBACK_MAX_BOX_AREA,
    ROBOT_FALLBACK_MIN_BOX_AREA,
    ROBOT_FALLBACK_MIN_CIRCULARITY,
    ROBOT_FALLBACK_MIN_FILL_RATIO,
    ROBOT_TEMPORAL_GATE_PX,
    ROBOT_TEMPORAL_IOU_GATE,
    ROBOT_TEMPORAL_MAX_AREA_RATIO,
    ROBOT_TEMPORAL_MIN_AREA_RATIO,
)


def _box_area(box: np.ndarray) -> float:
    return max(1.0, (box[2] - box[0]) * (box[3] - box[1]))


def _area_ratio(box: np.ndarray, reference: np.ndarray) -> float:
    return _box_area(box) / _box_area(reference)


def _confidence(det: sv.Detections, index: int) -> float:
    if det.confidence is None:
        return 0.0
    return float(det.confidence[index])


def robot_already_detected(cached_box: np.ndarray, fresh: sv.Detections) -> bool:
    if len(fresh) == 0:
        return False

    cached_center = cv_utils.center(cached_box)
    for i in range(len(fresh)):
        if cv_utils.iou(cached_box, fresh.xyxy[i]) > 0.2:
            return True
        fresh_center = cv_utils.center(fresh.xyxy[i])
        if np.hypot(cached_center[0] - fresh_center[0], cached_center[1] - fresh_center[1]) < 90.0:
            return True
    return False


def select_temporally_consistent_robots(
    fresh: sv.Detections,
    robot_cache: sv.Detections | None,
    *,
    max_robots: int,
) -> sv.Detections:
    if len(fresh) == 0:
        return fresh

    if robot_cache is None or len(robot_cache) == 0:
        return _top_confidence(fresh, max_robots)

    used_fresh: set[int] = set()
    selected_indices: list[int] = []

    for cache_index in range(len(robot_cache)):
        cached_box = robot_cache.xyxy[cache_index]
        cached_center = cv_utils.center(cached_box)
        best_index = None
        best_score = float("-inf")

        for fresh_index in range(len(fresh)):
            if fresh_index in used_fresh:
                continue

            box = fresh.xyxy[fresh_index]
            ratio = _area_ratio(box, cached_box)
            if ratio < ROBOT_TEMPORAL_MIN_AREA_RATIO or ratio > ROBOT_TEMPORAL_MAX_AREA_RATIO:
                continue

            center = cv_utils.center(box)
            distance = float(np.hypot(center[0] - cached_center[0], center[1] - cached_center[1]))
            iou = cv_utils.iou(cached_box, box)
            if iou < ROBOT_TEMPORAL_IOU_GATE and distance > ROBOT_TEMPORAL_GATE_PX:
                continue

            score = _confidence(fresh, fresh_index) + (iou * 2.0) - (distance / ROBOT_TEMPORAL_GATE_PX)
            if score > best_score:
                best_index = fresh_index
                best_score = score

        if best_index is not None:
            used_fresh.add(best_index)
            selected_indices.append(best_index)

    if len(robot_cache) < max_robots:
        remaining = [
            index
            for index in range(len(fresh))
            if index not in used_fresh
        ]
        remaining.sort(key=lambda index: _confidence(fresh, index), reverse=True)
        selected_indices.extend(remaining[: max_robots - len(selected_indices)])

    if not selected_indices:
        return sv.Detections.empty()

    return fresh[np.array(selected_indices[:max_robots])]


def filter_robot_roi_candidates(
    det: sv.Detections,
    cached_box: np.ndarray,
    *,
    roi_size_px: int,
) -> sv.Detections:
    if len(det) == 0:
        return det

    cached_center = cv_utils.center(cached_box)
    keep = []
    for i in range(len(det)):
        box = det.xyxy[i]
        ratio = _area_ratio(box, cached_box)
        center = cv_utils.center(box)
        distance = np.hypot(center[0] - cached_center[0], center[1] - cached_center[1])
        if 0.35 <= ratio <= 2.8 and distance <= roi_size_px * 0.45:
            keep.append(i)

    if not keep:
        return sv.Detections.empty()

    kept = det[np.array(keep)]
    best = int(np.argmax(kept.confidence)) if len(kept) > 1 else 0
    return kept[np.array([best])]


def foreground_robot_fallback(
    frame: np.ndarray,
    field_mask: np.ndarray | None,
    existing: sv.Detections | None = None,
    *,
    max_candidates: int,
) -> sv.Detections:
    if frame is None or frame.size == 0:
        return sv.Detections.empty()

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)

    green = ((hue >= 35) & (hue <= 95) & (sat >= 35) & (val >= 35)).astype(np.uint8) * 255
    field = _normalized_field_mask(field_mask, w, h) if field_mask is not None else green
    if field.size == 0 or int(np.count_nonzero(field)) == 0:
        return sv.Detections.empty()

    field = cv2.dilate(field, np.ones((9, 9), np.uint8), iterations=1)
    white_lines = ((sat <= 75) & (val >= 145)).astype(np.uint8) * 255
    foreground = cv2.bitwise_and(field, cv2.bitwise_not(green))
    foreground = cv2.bitwise_and(foreground, cv2.bitwise_not(white_lines))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1)

    contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return sv.Detections.empty()

    existing_boxes = existing.xyxy if existing is not None and len(existing) > 0 else np.empty((0, 4), dtype=float)
    candidates: list[tuple[float, np.ndarray]] = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        box_area = float(bw * bh)
        if box_area < ROBOT_FALLBACK_MIN_BOX_AREA or box_area > ROBOT_FALLBACK_MAX_BOX_AREA:
            continue

        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect > 2.25:
            continue

        contour_area = float(cv2.contourArea(contour))
        fill_ratio = contour_area / max(1.0, box_area)
        if fill_ratio < ROBOT_FALLBACK_MIN_FILL_RATIO:
            continue

        circularity = _circularity(contour)
        if circularity < ROBOT_FALLBACK_MIN_CIRCULARITY:
            continue

        box = np.array([x, y, x + bw, y + bh], dtype=float)
        center = cv_utils.center(box)
        if _overlaps_existing_robot(box, center, existing_boxes):
            continue

        score = (
            ROBOT_FALLBACK_CONFIDENCE
            + min(box_area / 12000.0, 1.0) * 0.16
            + min(fill_ratio, 1.0) * 0.18
            + min(circularity, 1.0) * 0.16
        )
        candidates.append((float(min(score, 0.88)), box))

    if not candidates:
        return sv.Detections.empty()

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max(1, max_candidates)]
    return sv.Detections(
        xyxy=np.array([box for _, box in selected], dtype=float),
        confidence=np.array([score for score, _ in selected], dtype=float),
        class_id=np.full(len(selected), CLASS_ROBOT, dtype=int),
        tracker_id=np.full(len(selected), -1, dtype=int),
    )


def _top_confidence(det: sv.Detections, max_items: int) -> sv.Detections:
    if len(det) <= max_items:
        return det
    if det.confidence is None:
        return det[np.arange(max_items)]
    top = np.argsort(-det.confidence)[:max_items]
    return det[top]


def _normalized_field_mask(field_mask: np.ndarray, width: int, height: int) -> np.ndarray:
    fm = field_mask
    if fm.ndim == 3:
        fm = np.any(fm, axis=0).astype(np.uint8)
    fm = (fm * 255).astype(np.uint8) if fm.max() <= 1 else fm.astype(np.uint8)
    fm = cv2.resize(fm, (width, height), interpolation=cv2.INTER_NEAREST)
    return (fm > 0).astype(np.uint8) * 255


def _circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def _overlaps_existing_robot(
    box: np.ndarray,
    center: tuple[float, float],
    existing_boxes: np.ndarray,
) -> bool:
    for existing_box in existing_boxes:
        if cv_utils.iou(box, existing_box) > 0.20:
            return True
        existing_center = cv_utils.center(existing_box)
        if np.hypot(center[0] - existing_center[0], center[1] - existing_center[1]) < 85.0:
            return True
    return False
