import numpy as np
import supervision as sv

from vision import cv_utils
from vision.segmentation_config import (
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


def _top_confidence(det: sv.Detections, max_items: int) -> sv.Detections:
    if len(det) <= max_items:
        return det
    if det.confidence is None:
        return det[np.arange(max_items)]
    top = np.argsort(-det.confidence)[:max_items]
    return det[top]
