import numpy as np
import supervision as sv

def iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU entre dos boxes xyxy."""
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)

def nms_detections(det: sv.Detections, iou_thresh: float) -> sv.Detections:
    """NMS simple por confianza descendente."""
    if len(det) == 0:
        return det
    order = np.argsort(-det.confidence)
    keep  = []
    suppressed = set()
    for idx in order:
        if idx in suppressed:
            continue
        keep.append(idx)
        for other in order:
            if other in suppressed or other == idx:
                continue
            if iou(det.xyxy[idx], det.xyxy[other]) > iou_thresh:
                suppressed.add(other)
    return det[np.array(keep)]

def center(xyxy: np.ndarray) -> tuple[float, float]:
    return ((xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2)
