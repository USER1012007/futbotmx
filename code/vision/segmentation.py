import cv2
import supervision as sv
import numpy as np
from trackers import ByteTrackTracker
from ultralytics.models.sam import SAM3SemanticPredictor
from domain.entities import FrameResult, Robot, Ball, make_pixel_point
from infra.configs import Config

# ── Constantes ──────────────────────────────────────────────────────────────
CLASS_ROBOT = 0
CLASS_BALL  = 1
CLASS_FIELD = 2

CONF_THRESHOLD   = 0.15        # más bajo para no perder bola naranja pequeña
MAX_ROBOTS       = 4
MAX_BALL_AGE     = 8
MAX_ROBOT_AGE    = 12
BALL_MIN_AREA    = 12          # bola naranja es MUY pequeña en imagen
BALL_MAX_AREA    = 2500
NMS_IOU_BALL     = 0.2
FIELD_CACHE_EVERY = 30
BALL_SEARCH_RADIUS_PX = 110.0
BALL_SEARCH_GROWTH_PX = 35.0
BALL_MAX_SEARCH_RADIUS_PX = 320.0
BALL_ROBOT_REJECT_RADIUS_PX = 58.0
BALL_ROBOT_REJECT_PADDING_PX = 14.0
BALL_MAX_ASPECT_RATIO = 1.7
BALL_MIN_CIRCULARITY = 0.55
BALL_MIN_ORANGE_PIXELS = 8
BALL_MIN_ORANGE_RATIO = 0.06
BALL_HSV_CONFIDENCE = 0.95

TEXT_PROMPTS = [
    "small wheeled soccer robot on green field",          # class 0
    "small orange ball on green surface",                 # class 1
    "flat green soccer field surface", # class 2
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU entre dos boxes xyxy."""
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def _nms_detections(det: sv.Detections, iou_thresh: float) -> sv.Detections:
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
            if _iou(det.xyxy[idx], det.xyxy[other]) > iou_thresh:
                suppressed.add(other)
    return det[np.array(keep)]


def _box_area(xyxy: np.ndarray) -> float:
    return max(0, xyxy[2]-xyxy[0]) * max(0, xyxy[3]-xyxy[1])


def _center(xyxy: np.ndarray):
    return ((xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2)


# ── Motor ────────────────────────────────────────────────────────────────────
class SegmentationEngine:
    def __init__(self, cfg: Config):
        model_path = str(cfg.BASE_DIR / "sam3.pt")
        overrides = dict(
            conf=CONF_THRESHOLD,
            task="segment",
            mode="predict",
            model=model_path,
            device="cuda",
            imgsz=640,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.cfg = cfg

        self.robot_tracker = ByteTrackTracker(
            lost_track_buffer=15,
            track_activation_threshold=0.15,
            minimum_iou_threshold=0.2,
        )
        self.ball_tracker = ByteTrackTracker(
            lost_track_buffer=10,
            track_activation_threshold=0.1,
            minimum_iou_threshold=0.1,
        )

        # Persistencia por clase
        self._robot_cache: sv.Detections | None = None   # última detección válida de robots
        self._ball_cache:  sv.Detections | None = None   # última detección válida de bola
        self._robot_age  = 0
        self._ball_age   = 0
        self._last_ball_center: tuple[float, float] | None = None
        self._last_ball_velocity: tuple[float, float] = (0.0, 0.0)

        self.last_positions: dict = {}
        self.cached_field_mask: np.ndarray | None = None
        self.frame_counter = 0

    # ── Persistencia por clase ────────────────────────────────────────────────
    def _update_robot_cache(self, det: sv.Detections):
        if len(det) > 0:
            self._robot_cache = det
            self._robot_age   = 0
        else:
            self._robot_age += 1

    def _update_ball_cache(self, det: sv.Detections):
        if len(det) > 0:
            self._ball_cache = det
            self._ball_age   = 0
        else:
            self._ball_age += 1

    def _get_robots(self, fresh: sv.Detections) -> sv.Detections:
        self._update_robot_cache(fresh)
        if len(fresh) > 0:
            return fresh
        if self._robot_cache is not None and self._robot_age <= MAX_ROBOT_AGE:
            return self._robot_cache
        return sv.Detections.empty()

    def _get_ball(self, fresh: sv.Detections, frame: np.ndarray) -> sv.Detections:
        # Filtrar por tamaño antes de persistir
        valid = _filter_ball_by_area(fresh)
        valid = _nms_detections(valid, NMS_IOU_BALL)

        # Fallback HSV si SAM no encontró bola
        if len(valid) == 0:
            valid = _hsv_ball_fallback(frame, self.cached_field_mask)

        self._update_ball_cache(valid)
        if len(valid) > 0:
            return valid
        if self._ball_cache is not None and self._ball_age <= MAX_BALL_AGE:
            return self._ball_cache
        return sv.Detections.empty()

    # ── Frame ─────────────────────────────────────────────────────────────────
    def _get_ball_candidates(
        self,
        fresh: sv.Detections,
        frame: np.ndarray,
        robot_boxes: np.ndarray,
    ) -> sv.Detections:
        valid = _filter_ball_by_area(fresh)
        valid = _filter_ball_by_aspect_ratio(valid)
        valid = _filter_ball_by_orange_support(valid, frame)
        valid = _filter_ball_against_robots(valid, robot_boxes)
        valid = _nms_detections(valid, NMS_IOU_BALL)

        hsv = _hsv_ball_fallback(frame, self.cached_field_mask, robot_boxes)
        if len(hsv) > 0:
            return self._gate_ball_candidates(hsv)

        return self._gate_ball_candidates(valid)

    def _get_tracked_ball(self, tracked: sv.Detections) -> sv.Detections:
        self._update_ball_cache(tracked)
        if len(tracked) > 0:
            return tracked
        if self._ball_cache is not None and self._ball_age <= MAX_BALL_AGE:
            return self._ball_cache
        return sv.Detections.empty()

    def _gate_ball_candidates(self, det: sv.Detections) -> sv.Detections:
        if len(det) == 0 or self._last_ball_center is None:
            return det

        pred_x = self._last_ball_center[0] + self._last_ball_velocity[0]
        pred_y = self._last_ball_center[1] + self._last_ball_velocity[1]
        radius = min(
            BALL_MAX_SEARCH_RADIUS_PX,
            BALL_SEARCH_RADIUS_PX + self._ball_age * BALL_SEARCH_GROWTH_PX,
        )

        scored = []
        for i in range(len(det)):
            cx, cy = _center(det.xyxy[i])
            distance = np.hypot(cx - pred_x, cy - pred_y)
            if distance <= radius:
                confidence = float(det.confidence[i]) if det.confidence is not None else 0.0
                scored.append((distance - confidence * 20.0, i))

        if not scored:
            return sv.Detections.empty()

        best_idx = min(scored, key=lambda item: item[0])[1]
        return det[np.array([best_idx])]

    def _remember_ball_center(self, center: tuple[float, float]) -> None:
        if self._last_ball_center is not None:
            self._last_ball_velocity = (
                center[0] - self._last_ball_center[0],
                center[1] - self._last_ball_center[1],
            )
        self._last_ball_center = center

    def process_frame(self, frame: cv2.typing.MatLike, frame_id: int) -> FrameResult:
        self.predictor.set_image(frame)
        results    = self.predictor(text=TEXT_PROMPTS)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filtro global de confianza
        detections = detections[detections.confidence > CONF_THRESHOLD]

        # Split por clase
        robots_fresh = detections[detections.class_id == CLASS_ROBOT]
        ball_fresh   = detections[detections.class_id == CLASS_BALL]
        field_fresh  = detections[detections.class_id == CLASS_FIELD]

        # Limitar robots a MAX_ROBOTS por confianza
        if len(robots_fresh) > MAX_ROBOTS:
            top = np.argsort(-robots_fresh.confidence)[:MAX_ROBOTS]
            robots_fresh = robots_fresh[top]

        # Campo (cacheado). La mascara tambien ayuda a descartar manos/fondo en HSV.
        if self.frame_counter % FIELD_CACHE_EVERY == 0 or self.cached_field_mask is None:
            self.cached_field_mask = _build_field_mask(field_fresh, self.cached_field_mask)

        # Persistencia por clase
        robots_fresh = self.robot_tracker.update(robots_fresh) if len(robots_fresh) > 0 else sv.Detections.empty()
        robots_det = self._get_robots(robots_fresh)
        robot_boxes = robots_det.xyxy if len(robots_det) > 0 else np.empty((0, 4), dtype=float)
        ball_candidates = self._get_ball_candidates(ball_fresh, frame, robot_boxes)
        ball_tracked = self.ball_tracker.update(ball_candidates) if len(ball_candidates) > 0 else sv.Detections.empty()
        ball_det = self._get_tracked_ball(ball_tracked)

        self.frame_counter += 1

        # ── Construir entidades ───────────────────────────────────────────────
        robots = []
        for i in range(len(robots_det)):
            # ByteTrack garantiza que tracker_id sea un entero único
            tid = int(robots_det.tracker_id[i]) if robots_det.tracker_id is not None else -1
            if tid == -1:
                continue
            
            xyxy = robots_det.xyxy[i]
            pos  = make_pixel_point(*_center(xyxy))
            robots.append(Robot(
                id=f"robot_{tid}",
                tracker_id=tid,
                team_id="unknown",
                position_pixel=pos,
            ))
            self.last_positions[tid] = (pos.x, pos.y)

        best_ball = None
        if len(ball_det) > 0:
            best_idx = int(np.argmax(ball_det.confidence))
            tid  = int(ball_det.tracker_id[best_idx]) if ball_det.tracker_id is not None else -1
            xyxy = ball_det.xyxy[best_idx]
            pos  = make_pixel_point(*_center(xyxy))
            best_ball = Ball(
                id="ball",
                tracker_id=tid,
                position_pixel=pos,
            )
            self._remember_ball_center((pos.x, pos.y))
            self.last_positions["ball"] = (pos.x, pos.y)

        return FrameResult(
            frame_id=frame_id,
            robots=robots,
            ball=best_ball,
            field_mask=self.cached_field_mask,
        )


# ── Utilidades de módulo ──────────────────────────────────────────────────────
def _filter_ball_by_area(det: sv.Detections) -> sv.Detections:
    if len(det) == 0:
        return det
    areas = np.array([_box_area(det.xyxy[i]) for i in range(len(det))])
    mask  = (areas >= BALL_MIN_AREA) & (areas <= BALL_MAX_AREA)
    return det[mask]


def _filter_ball_by_aspect_ratio(det: sv.Detections) -> sv.Detections:
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


def _filter_ball_by_orange_support(det: sv.Detections, frame: np.ndarray) -> sv.Detections:
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


def _filter_ball_against_robots(det: sv.Detections, robot_boxes: np.ndarray) -> sv.Detections:
    if len(det) == 0 or len(robot_boxes) == 0:
        return det

    keep = []
    for i in range(len(det)):
        if not _is_near_robot(_center(det.xyxy[i]), robot_boxes):
            keep.append(i)

    return det[np.array(keep)] if keep else sv.Detections.empty()


def _build_field_mask(
    field_det: sv.Detections,
    previous: np.ndarray | None,
) -> np.ndarray | None:
    if len(field_det) == 0 or field_det.mask is None:
        return previous
    raw      = field_det.mask.astype(np.uint8)
    combined = np.any(raw, axis=0).astype(np.uint8)
    kernel   = np.ones((15, 15), np.uint8)
    return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)


# ── Fallback HSV para bola naranja ───────────────────────────────────────────
# Rango naranja en HSV (campo turquesa no interfiere)
_HSV_ORANGE_LO = np.array([0,  100, 90], dtype=np.uint8)
_HSV_ORANGE_HI = np.array([7, 255, 255], dtype=np.uint8)


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


def _hsv_ball_fallback(
    frame: np.ndarray,
    field_mask: np.ndarray | None,
    robot_boxes: np.ndarray | None = None,
) -> sv.Detections:
    """
    Detecta bola naranja por color cuando SAM falla.
    Retorna sv.Detections con 0 o 1 detección.
    """
    mask = _orange_mask(frame)

    # Solo dentro del campo si tenemos máscara
    if field_mask is not None:
        fm = (field_mask * 255).astype(np.uint8) if field_mask.max() <= 1 else field_mask
        fm = cv2.resize(fm, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = cv2.bitwise_and(mask, fm)

    # Morfología para limpiar ruido
    k    = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return sv.Detections.empty()

    # Elegir contorno más circular dentro de rango de área
    best_box  = None
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
        xyxy        = best_box[np.newaxis],
        confidence  = np.array([BALL_HSV_CONFIDENCE]),
        class_id    = np.array([CLASS_BALL]),
        tracker_id  = np.array([-1]),           # tracker asignará ID
    )
