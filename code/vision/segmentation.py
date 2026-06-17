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

CONF_THRESHOLD   = 0.20        # más bajo para no perder bola naranja pequeña
MAX_ROBOTS       = 3
MAX_BALL_AGE     = 8
MAX_ROBOT_AGE    = 12
BALL_MIN_AREA    = 15          # bola naranja es MUY pequeña en imagen
BALL_MAX_AREA    = 2500
NMS_IOU_BALL     = 0.3
FIELD_CACHE_EVERY = 30

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

        self.tracker = ByteTrackTracker(
            lost_track_buffer=15,
            track_activation_threshold=0.15,
            minimum_iou_threshold=0.2,
        )

        # Persistencia por clase
        self._robot_cache: sv.Detections | None = None   # última detección válida de robots
        self._ball_cache:  sv.Detections | None = None   # última detección válida de bola
        self._robot_age  = 0
        self._ball_age   = 0

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
    def process_frame(self, frame: cv2.typing.MatLike, frame_id: int) -> FrameResult:
        self.predictor.set_image(frame)
        results    = self.predictor(text=TEXT_PROMPTS)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filtro global de confianza
        detections = detections[detections.confidence > CONF_THRESHOLD]

        # Tracking (sobre todas las clases juntas, mantiene IDs estables)
        detections = self.tracker.update(detections)

        # Split por clase
        robots_fresh = detections[detections.class_id == CLASS_ROBOT]
        ball_fresh   = detections[detections.class_id == CLASS_BALL]
        field_fresh  = detections[detections.class_id == CLASS_FIELD]

        # Limitar robots a MAX_ROBOTS por confianza
        if len(robots_fresh) > MAX_ROBOTS:
            top = np.argsort(-robots_fresh.confidence)[:MAX_ROBOTS]
            robots_fresh = robots_fresh[top]

        # Persistencia por clase
        robots_det = self._get_robots(robots_fresh)
        ball_det   = self._get_ball(ball_fresh, frame)

        # Campo (cacheado)
        if self.frame_counter % FIELD_CACHE_EVERY == 0 or self.cached_field_mask is None:
            self.cached_field_mask = _build_field_mask(field_fresh, self.cached_field_mask)

        self.frame_counter += 1

        # ── Construir entidades ───────────────────────────────────────────────
        robots = []
        for i in range(len(robots_det)):
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
                id=f"ball_{tid}",
                tracker_id=tid,
                position_pixel=pos,
            )
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
_HSV_ORANGE_LO = np.array([5,  120, 120], dtype=np.uint8)
_HSV_ORANGE_HI = np.array([25, 255, 255], dtype=np.uint8)

def _hsv_ball_fallback(frame: np.ndarray, field_mask: np.ndarray | None) -> sv.Detections:
    """
    Detecta bola naranja por color cuando SAM falla.
    Retorna sv.Detections con 0 o 1 detección.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _HSV_ORANGE_LO, _HSV_ORANGE_HI)

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
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim ** 2)
        if circularity > best_circ:
            best_circ = circularity
            x, y, w, h = cv2.boundingRect(cnt)
            best_box = np.array([x, y, x+w, y+h], dtype=float)

    if best_box is None or best_circ < 0.3:   # muy no-circular → no es bola
        return sv.Detections.empty()

    return sv.Detections(
        xyxy        = best_box[np.newaxis],
        confidence  = np.array([0.6]),          # confianza fija moderada
        class_id    = np.array([CLASS_BALL]),
        tracker_id  = np.array([-1]),           # tracker asignará ID
    )
