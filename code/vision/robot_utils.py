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
    ROBOT_FALLBACK_SPLIT_MIN_AREA,
    ROBOT_FALLBACK_SPLIT_MIN_SEPARATION_PX,
    ROBOT_DUPLICATE_CENTER_GATE_PX,
    ROBOT_DUPLICATE_IOU_GATE,
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
        if cv_utils.iou(cached_box, fresh.xyxy[i]) > ROBOT_DUPLICATE_IOU_GATE:
            return True
        fresh_center = cv_utils.center(fresh.xyxy[i])
        if (
            np.hypot(cached_center[0] - fresh_center[0], cached_center[1] - fresh_center[1])
            < ROBOT_DUPLICATE_CENTER_GATE_PX
        ):
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
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

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
        score = (
            ROBOT_FALLBACK_CONFIDENCE
            + min(box_area / 12000.0, 1.0) * 0.16
            + min(fill_ratio, 1.0) * 0.18
            + min(circularity, 1.0) * 0.16
        )
        split_candidates = _split_large_robot_candidate(
            foreground,
            box,
            existing_boxes,
            base_score=float(min(score + 0.03, 0.86)),
        )
        if split_candidates:
            candidates.extend(split_candidates)
            continue

        if _overlaps_existing_robot(box, center, existing_boxes):
            continue

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
        if cv_utils.iou(box, existing_box) > ROBOT_DUPLICATE_IOU_GATE:
            return True
        existing_center = cv_utils.center(existing_box)
        if (
            np.hypot(center[0] - existing_center[0], center[1] - existing_center[1])
            < ROBOT_DUPLICATE_CENTER_GATE_PX
        ):
            return True
    return False


def _split_large_robot_candidate(
    foreground: np.ndarray,
    box: np.ndarray,
    existing_boxes: np.ndarray,
    *,
    base_score: float,
) -> list[tuple[float, np.ndarray]]:
    x1, y1, x2, y2 = box.astype(int)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    area = width * height
    if area < ROBOT_FALLBACK_SPLIT_MIN_AREA:
        return []

    aspect = max(width / max(height, 1), height / max(width, 1))
    if aspect < 1.35:
        return []

    crop = foreground[y1:y2, x1:x2]
    if crop.size == 0:
        return []

    split_boxes = _projection_split_boxes(crop, x1, y1)
    if len(split_boxes) < 2:
        return []

    centers = [cv_utils.center(split_box) for split_box in split_boxes[:2]]
    separation = float(np.hypot(centers[0][0] - centers[1][0], centers[0][1] - centers[1][1]))
    if separation < ROBOT_FALLBACK_SPLIT_MIN_SEPARATION_PX:
        return []

    selected: list[tuple[float, np.ndarray]] = []
    for split_box in split_boxes[:2]:
        sx1, sy1, sx2, sy2 = split_box
        split_area = max(1.0, float((sx2 - sx1) * (sy2 - sy1)))
        if split_area < ROBOT_FALLBACK_MIN_BOX_AREA:
            continue
        center = cv_utils.center(split_box)
        if _overlaps_existing_robot(split_box, center, existing_boxes):
            continue
        selected.append((base_score, split_box.astype(float)))

    return selected


def _projection_split_boxes(crop: np.ndarray, offset_x: int, offset_y: int) -> list[np.ndarray]:
    height, width = crop.shape[:2]
    if width >= height:
        projection = np.count_nonzero(crop, axis=0)
        ranges = _foreground_ranges(projection, min_gap=6)
        if len(ranges) < 2:
            ranges = _valley_split_ranges(projection)
        boxes = [
            _tight_box(crop[:, start:end], offset_x + start, offset_y)
            for start, end in ranges
        ]
    else:
        projection = np.count_nonzero(crop, axis=1)
        ranges = _foreground_ranges(projection, min_gap=6)
        if len(ranges) < 2:
            ranges = _valley_split_ranges(projection)
        boxes = [
            _tight_box(crop[start:end, :], offset_x, offset_y + start)
            for start, end in ranges
        ]

    boxes = [box for box in boxes if box is not None]
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes


def _foreground_ranges(projection: np.ndarray, *, min_gap: int) -> list[tuple[int, int]]:
    active = projection > max(2, int(projection.max() * 0.08))
    ranges: list[tuple[int, int]] = []
    start = None
    gap = 0

    for idx, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = idx
            gap = 0
            continue
        if start is None:
            continue
        gap += 1
        if gap >= min_gap:
            end = idx - gap + 1
            if end - start >= 10:
                ranges.append((start, end))
            start = None
            gap = 0

    if start is not None and len(active) - start >= 10:
        ranges.append((start, len(active)))
    return ranges


def _valley_split_ranges(projection: np.ndarray) -> list[tuple[int, int]]:
    if projection.size < 40 or int(projection.max()) <= 0:
        return []

    active_indices = np.where(projection > max(2, int(projection.max() * 0.06)))[0]
    if active_indices.size < 30:
        return []

    start = int(active_indices.min())
    end = int(active_indices.max()) + 1
    if end - start < 40:
        return []

    smooth = np.convolve(projection.astype(float), np.ones(9) / 9.0, mode="same")
    left_guard = start + max(12, int((end - start) * 0.25))
    right_guard = end - max(12, int((end - start) * 0.25))
    if right_guard <= left_guard:
        return []

    split = int(left_guard + np.argmin(smooth[left_guard:right_guard]))
    left_peak = float(smooth[start:split].max()) if split > start else 0.0
    right_peak = float(smooth[split:end].max()) if end > split else 0.0
    valley = float(smooth[split])

    if left_peak <= 0 or right_peak <= 0:
        return []
    if valley > min(left_peak, right_peak) * 0.78:
        return []
    if split - start < 14 or end - split < 14:
        return []

    return [(start, split), (split, end)]


def _tight_box(crop: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray | None:
    ys, xs = np.where(crop > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([
        offset_x + int(xs.min()),
        offset_y + int(ys.min()),
        offset_x + int(xs.max()) + 1,
        offset_y + int(ys.max()) + 1,
    ], dtype=float)
