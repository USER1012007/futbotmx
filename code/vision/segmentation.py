import cv2
import supervision as sv
import numpy as np
from trackers import ByteTrackTracker
from ultralytics.models.sam import SAM3SemanticPredictor
from domain.entities import FrameResult, Robot, Ball, make_pixel_point
from infra.configs import Config
from vision import ball_utils, cv_utils

# ── Constantes ──────────────────────────────────────────────────────────────
CLASS_ROBOT = 0
CLASS_BALL  = 1
CLASS_FIELD = 2

CONF_THRESHOLD   = 0.15        # más bajo para no perder bola naranja pequena
MAX_ROBOTS       = 4
MAX_BALL_AGE     = 8
MAX_ROBOT_AGE    = 12
NMS_IOU_BALL     = 0.2
FIELD_CACHE_EVERY = 30
BALL_SEARCH_RADIUS_PX = 110.0
BALL_SEARCH_GROWTH_PX = 35.0
BALL_MAX_SEARCH_RADIUS_PX = 320.0

TEXT_PROMPTS = [
    "small wheeled soccer robot on green field",          # class 0
    "small orange ball on green surface",                 # class 1
    "flat green soccer field surface", # class 2
]

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
        valid = ball_utils.filter_ball_by_area(fresh)
        valid = cv_utils.nms_detections(valid, NMS_IOU_BALL)

        # Fallback HSV si SAM no encontró bola
        if len(valid) == 0:
            valid = ball_utils.hsv_ball_fallback(frame, self.cached_field_mask)

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
        valid = ball_utils.filter_ball_by_area(fresh)
        valid = ball_utils.filter_ball_by_aspect_ratio(valid)
        valid = ball_utils.filter_ball_by_orange_support(valid, frame)
        valid = ball_utils.filter_ball_against_robots(valid, robot_boxes)
        valid = cv_utils.nms_detections(valid, NMS_IOU_BALL)

        hsv = ball_utils.hsv_ball_fallback(frame, self.cached_field_mask, robot_boxes)
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
            cx, cy = cv_utils.center(det.xyxy[i])
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
            pos  = make_pixel_point(*cv_utils.center(xyxy))
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
            pos  = make_pixel_point(*cv_utils.center(xyxy))
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
