import cv2
import numpy as np
import supervision as sv

# Constantes exportadas para que el motor las use
BALL_MIN_AREA = 12
BALL_MAX_AREA = 2500
BALL_MAX_ASPECT_RATIO = 1.7
BALL_MIN_CIRCULARITY = 0.55
BALL_MIN_ORANGE_PIXELS = 8
BALL_MIN_ORANGE_RATIO = 0.06
BALL_HSV_CONFIDENCE = 0.95
BALL_ROBOT_REJECT_RADIUS_PX = 58.0
BALL_ROBOT_REJECT_PADDING_PX = 14.0

_HSV_ORANGE_LO = np.array([0, 100, 90], dtype=np.uint8)
_HSV_ORANGE_HI = np.array([7, 255, 255], dtype=np.uint8)

def _box_area(xyxy: np.ndarray) -> float:
    return max(0, xyxy[2] - xyxy[0]) * max(0, xyxy[3] - xyxy[1])

def _center(xyxy: np.ndarray):
    return ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)

def _orange_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, _HSV_ORANGE_LO, _HSV_ORANGE_HI)

def _clip_box(xyxy: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(width, int(np.floor(xyxy[0]))))
    y0 = max(0, min(height, int(np.floor(xyxy[1]))))
    x1 = max(0, min(width, int(np.ceil(xyxy[2]))))
    y1 = max(0, min(height, int(np.ceil(xyxy[3]))))
    return x0, y0, x1, y1

def _is_near_robot(center: tuple[float, float], robot_boxes: np.ndarray) -> bool:
    cx, cy = center
    for box in robot_boxes:
        x0, y0, x1, y1 = box
        expanded_x0 = x0 - BALL_ROBOT_REJECT_PADDING_PX
        expanded_y0 = y0 - BALL_ROBOT_REJECT_PADDING_PX
        expanded_x1 = x1 + BALL_ROBOT_REJECT_PADDING_PX
        expanded_y1 = y1 + BALL_ROBOT_REJECT_PADDING_PX
        if expanded_x0 <= cx <= expanded_x1 and expanded_y0 <= cy <= expanded_y1:
            return True
        rx, ry = _center(box)
        if np.hypot(cx - rx, cy - ry) < BALL_ROBOT_REJECT_RADIUS_PX:
            return True
    return False

def filter_ball_by_area(det: sv.Detections) -> sv.Detections:
    if len(det) == 0:
        return det
    areas = np.array([_box_area(det.xyxy[i]) for i in range(len(det))])
    mask = (areas >= BALL_MIN_AREA) & (areas <= BALL_MAX_AREA)
    return det[mask]

def filter_ball_by_aspect_ratio(det: sv.Detections) -> sv.Detections:
    if len(det) == 0:
        return det
    keep = []
    for i in range(len(det)):
        x0, y0, x1, y1 = det.xyxy[i]
        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)
        ratio = max(width / height, height / width)
        if ratio <= BALL_MAX_ASPECT_RATIO:
            keep.append(i)
    return det[np.array(keep)] if keep else sv.Detections.empty()

def filter_ball_by_orange_support(det: sv.Detections, frame: np.ndarray) -> sv.Detections:
    if len(det) == 0:
        return det

    orange_mask = _orange_mask(frame)
    keep = []
    for i in range(len(det)):
        x0, y0, x1, y1 = _clip_box(det.xyxy[i], frame.shape[1], frame.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        roi = orange_mask[y0:y1, x0:x1]
        orange_pixels = int(np.count_nonzero(roi))
        orange_ratio = orange_pixels / max(1, roi.size)
        if orange_pixels >= BALL_MIN_ORANGE_PIXELS and orange_ratio >= BALL_MIN_ORANGE_RATIO:
            keep.append(i)
    return det[np.array(keep)] if keep else sv.Detections.empty()

def filter_ball_against_robots(det: sv.Detections, robot_boxes: np.ndarray) -> sv.Detections:
    if len(det) == 0 or len(robot_boxes) == 0:
        return det
    keep = []
    for i in range(len(det)):
        if not _is_near_robot(_center(det.xyxy[i]), robot_boxes):
            keep.append(i)
    return det[np.array(keep)] if keep else sv.Detections.empty()

def hsv_ball_fallback(
    frame: np.ndarray,
    field_mask: np.ndarray | None,
    robot_boxes: np.ndarray | None = None,
) -> sv.Detections:
    mask = _orange_mask(frame)
    if field_mask is not None:
        fm = (field_mask * 255).astype(np.uint8) if field_mask.max() <= 1 else field_mask
        fm = cv2.resize(fm, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = cv2.bitwise_and(mask, fm)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return sv.Detections.empty()
    best_box = None
    best_circ = -1.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (BALL_MIN_AREA <= area <= BALL_MAX_AREA):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        if aspect_ratio > BALL_MAX_ASPECT_RATIO:
            continue
        center = (x + w / 2.0, y + h / 2.0)
        if robot_boxes is not None and len(robot_boxes) > 0 and _is_near_robot(center, robot_boxes):
            continue
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim ** 2)
        if circularity > best_circ:
            best_circ = circularity
            x, y, w, h = cv2.boundingRect(cnt)
            best_box = np.array([x, y, x+w, y+h], dtype=float)
    if best_box is None or best_circ < BALL_MIN_CIRCULARITY:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=best_box[np.newaxis],
        confidence=np.array([BALL_HSV_CONFIDENCE]),
        class_id=np.array([1]), # CLASS_BALL
        tracker_id=np.array([-1]),
    )
