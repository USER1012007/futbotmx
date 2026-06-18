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

CONF_THRESHOLD   = 0.25        # más bajo para no perder bola naranja pequena
MAX_ROBOTS       = 4
MAX_BALL_AGE     = 10
MAX_ROBOT_AGE    = 12
NMS_IOU_BALL     = 0.0
FIELD_CACHE_EVERY = 30
ROBOT_ROI_SIZE_PX = 320
ROBOT_ROI_INFER_SIZE_PX = 640
BALL_ROI_SIZE_PX = 256
BALL_ROI_INFER_SIZE_PX = 640
BALL_SEARCH_RADIUS_PX = 80.0
BALL_SEARCH_GROWTH_PX = 15.0
BALL_MAX_SEARCH_RADIUS_PX = 200.0
BALL_RECOVERY_AFTER_FRAMES = MAX_BALL_AGE + 1
BALL_CONFIDENCE_WEIGHT = 90.0
BALL_ORANGE_WEIGHT = 70.0
BALL_DISTANCE_WEIGHT = 0.85
BALL_RECENT_REJECT_MARGIN_PX = 30.0

TEXT_PROMPTS = [
    "small wheeled soccer robot on green field",          # class 0
    "small bright orange circular ball on green floor, vivid saturated orange dot, overhead view",
    "flat green soccer field surface", # class 2
]

ROI_BALL_PROMPTS = [
    "bright orange ball",
    "small orange circular ball",
    "orange soccer ball",
    "vivid orange dot",
]

ROI_ROBOT_PROMPTS = [
    "wheeled soccer robot",
    "small soccer robot",
    "robot on green soccer field",
    "round wheeled robot",
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
            lost_track_buffer=8,
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
        self.last_ball_debug: dict = {}

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
        include_aux_sources: bool = True,
    ) -> sv.Detections:
        # 1. Candidatos de SAM
        sam_candidates = ball_utils.filter_ball_by_area(fresh)
        sam_candidates = ball_utils.filter_ball_by_aspect_ratio(sam_candidates)
        sam_candidates = ball_utils.filter_ball_by_orange_support(sam_candidates, frame)
        sam_candidates = ball_utils.filter_ball_by_orange_signature(sam_candidates, frame)
        sam_candidates = ball_utils.filter_ball_by_field_context(
            sam_candidates,
            self.cached_field_mask,
            frame.shape,
        )
        sam_candidates = ball_utils.filter_ball_against_robots(sam_candidates, robot_boxes)

        # 2. Candidatos auxiliares. En imagen completa usamos HSV/template como
        # respaldo; en ROI ya vienen fuentes locales para no contaminar con manos
        # o falsos positivos de otra zona.
        hsv_candidates = sv.Detections.empty()
        template_candidates = sv.Detections.empty()
        if include_aux_sources:
            hsv_candidates = ball_utils.hsv_ball_fallback(
                frame,
                self.cached_field_mask,
                robot_boxes,
                max_candidates=5,
            )
            if self._last_ball_center is not None:
                template_candidates = ball_utils.template_match_ball(
                    frame,
                    self.cached_field_mask,
                    search_center=self._predicted_ball_center(),
                    search_radius=self._search_radius(),
                    robot_boxes=robot_boxes,
                )

        # 3. Combinar conjuntos y aplicar NMS.
        all_candidates = self._combine_detections(
            [sam_candidates, hsv_candidates, template_candidates]
        )
        all_candidates = cv_utils.nms_detections(all_candidates, 0.3)
        all_candidates = ball_utils.filter_ball_by_field_context(
            all_candidates,
            self.cached_field_mask,
            frame.shape,
        )
        all_candidates = ball_utils.filter_ball_by_orange_signature(all_candidates, frame)

        # 4. Elegir un solo candidato con restricciones fisicas/temporales antes
        # de ByteTrack. Esto evita saltos imposibles al otro lado de la cancha.
        return self._select_ball_candidate(all_candidates, frame)

    def _get_tracked_ball(self, tracked: sv.Detections) -> sv.Detections:
        self._update_ball_cache(tracked)
        if len(tracked) > 0:
            return tracked
        if self._ball_cache is not None and self._ball_age <= MAX_BALL_AGE:
            return self._ball_cache
        return sv.Detections.empty()

    def _select_ball_candidate(self, det: sv.Detections, frame: np.ndarray) -> sv.Detections:
        if len(det) == 0:
            self.last_ball_debug = {
                "state": self._ball_state(),
                "candidate_count": 0,
                "accepted": False,
                "reason": "no_candidates",
            }
            return det

        state = self._ball_state()
        predicted = self._predicted_ball_center()
        radius = self._search_radius()
        locked_to_recent_track = predicted is not None and state in {"locked", "lost"}

        scored = []
        for i in range(len(det)):
            cx, cy = cv_utils.center(det.xyxy[i])
            confidence = float(det.confidence[i]) if det.confidence is not None else 0.0
            signature = ball_utils.orange_signature_for_box(det.xyxy[i], frame)
            orange_ratio = signature["orange_ratio"]
            circularity = signature["circularity"]
            field_ratio = ball_utils.field_context_ratio_for_box(
                det.xyxy[i],
                self.cached_field_mask,
                frame.shape,
            )
            distance = 0.0
            if predicted is not None:
                distance = float(np.hypot(cx - predicted[0], cy - predicted[1]))

            if field_ratio < ball_utils.BALL_MIN_FIELD_NEIGHBOR_RATIO:
                continue
            if (
                signature["orange_pixels"] < ball_utils.BALL_MIN_ORANGE_PIXELS
                or orange_ratio < ball_utils.BALL_MIN_SIGNATURE_RATIO
                or circularity < ball_utils.BALL_MIN_SIGNATURE_CIRCULARITY
            ):
                continue
            if locked_to_recent_track and distance > radius + BALL_RECENT_REJECT_MARGIN_PX:
                continue

            score = (
                confidence * BALL_CONFIDENCE_WEIGHT
                + orange_ratio * BALL_ORANGE_WEIGHT
                + circularity * 25.0
                - distance * BALL_DISTANCE_WEIGHT
            )
            scored.append((score, i, distance, confidence, orange_ratio, circularity, field_ratio))

        if not scored:
            self.last_ball_debug = {
                "state": state,
                "candidate_count": len(det),
                "accepted": False,
                "reason": "all_candidates_outside_motion_gate",
                "predicted_center": predicted,
                "radius_px": radius,
            }
            return sv.Detections.empty()

        (
            best_score,
            best_idx,
            best_distance,
            best_confidence,
            best_orange_ratio,
            best_circularity,
            best_field_ratio,
        ) = max(
            scored,
            key=lambda item: item[0],
        )
        self.last_ball_debug = {
            "state": state,
            "candidate_count": len(det),
            "accepted": True,
            "reason": "best_score",
            "score": float(best_score),
            "distance_px": float(best_distance),
            "confidence": float(best_confidence),
            "orange_ratio": float(best_orange_ratio),
            "orange_circularity": float(best_circularity),
            "field_context_ratio": float(best_field_ratio),
            "predicted_center": predicted,
            "radius_px": radius,
        }
        return det[np.array([best_idx])]

    # Alias temporal para codigo diagnostico o pruebas que llamen al nombre viejo.
    def _gate_ball_candidates(self, det: sv.Detections) -> sv.Detections:
        return self._select_ball_candidate(det, np.zeros((1, 1, 3), dtype=np.uint8))

    def _predicted_ball_center(self) -> tuple[float, float] | None:
        if self._last_ball_center is None:
            return None
        return (
            self._last_ball_center[0] + self._last_ball_velocity[0],
            self._last_ball_center[1] + self._last_ball_velocity[1],
        )

    def _search_radius(self) -> float:
        return min(
            BALL_MAX_SEARCH_RADIUS_PX,
            BALL_SEARCH_RADIUS_PX + self._ball_age * BALL_SEARCH_GROWTH_PX,
        )

    def _ball_state(self) -> str:
        if self._last_ball_center is None:
            return "search"
        if self._ball_age <= 0:
            return "locked"
        if self._ball_age < BALL_RECOVERY_AFTER_FRAMES:
            return "lost"
        return "search"

    def _remember_ball_center(self, center: tuple[float, float]) -> None:
        if self._last_ball_center is not None:
            self._last_ball_velocity = (
                center[0] - self._last_ball_center[0],
                center[1] - self._last_ball_center[1],
            )
        self._last_ball_center = center

    @staticmethod
    def _combine_detections(detections: list[sv.Detections]) -> sv.Detections:
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
                det.class_id if det.class_id is not None else np.full(len(det), CLASS_BALL)
                for det in non_empty
            ]),
        )


    def _infer_full_frame(self, frame: np.ndarray) -> sv.Detections:
        self.predictor.set_image(frame)
        results = self.predictor(text=TEXT_PROMPTS)[0]
        return sv.Detections.from_ultralytics(results)

    def _infer_ball_roi(self, frame: np.ndarray) -> tuple[sv.Detections, tuple[int, int, int, int] | None]:
        if self._last_ball_center is None:
            return sv.Detections.empty(), None

        x1, y1, x2, y2 = self._roi_bounds(
            frame.shape[1],
            frame.shape[0],
            self._last_ball_center,
            BALL_ROI_SIZE_PX,
        )
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return sv.Detections.empty(), (x1, y1, x2, y2)

        crop_h, crop_w = crop.shape[:2]
        sam_crop = cv2.resize(
            crop,
            (BALL_ROI_INFER_SIZE_PX, BALL_ROI_INFER_SIZE_PX),
            interpolation=cv2.INTER_CUBIC,
        )

        self.predictor.set_image(sam_crop)
        roi_results = self.predictor(text=ROI_BALL_PROMPTS)[0]
        roi_dets = sv.Detections.from_ultralytics(roi_results)
        roi_dets = self._scale_roi_detections_to_frame(
            roi_dets,
            offset=(x1, y1),
            roi_size=(crop_w, crop_h),
            infer_size=(BALL_ROI_INFER_SIZE_PX, BALL_ROI_INFER_SIZE_PX),
            class_id=CLASS_BALL,
        )

        hsv_dets = ball_utils.hsv_ball_fallback(
            crop,
            None,
            max_candidates=3,
        )
        hsv_dets = self._offset_roi_detections_to_frame(hsv_dets, offset=(x1, y1))

        template_dets = ball_utils.template_match_ball(
            crop,
            None,
            search_center=(crop_w / 2.0, crop_h / 2.0),
            search_radius=BALL_ROI_SIZE_PX / 2.0,
        )
        template_dets = self._offset_roi_detections_to_frame(template_dets, offset=(x1, y1))

        return self._combine_detections([roi_dets, hsv_dets, template_dets]), (x1, y1, x2, y2)

    def _infer_missing_robot_rois(self, frame: np.ndarray, fresh: sv.Detections) -> sv.Detections:
        if self._robot_cache is None or len(self._robot_cache) == 0:
            return sv.Detections.empty()

        roi_detections = []
        for i in range(len(self._robot_cache)):
            cached_box = self._robot_cache.xyxy[i]
            if self._robot_already_detected(cached_box, fresh):
                continue

            center = cv_utils.center(cached_box)
            x1, y1, x2, y2 = self._roi_bounds(
                frame.shape[1],
                frame.shape[0],
                center,
                ROBOT_ROI_SIZE_PX,
            )
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]
            sam_crop = cv2.resize(
                crop,
                (ROBOT_ROI_INFER_SIZE_PX, ROBOT_ROI_INFER_SIZE_PX),
                interpolation=cv2.INTER_CUBIC,
            )
            self.predictor.set_image(sam_crop)
            roi_results = self.predictor(text=ROI_ROBOT_PROMPTS)[0]
            roi_dets = sv.Detections.from_ultralytics(roi_results)
            roi_dets = self._scale_roi_detections_to_frame(
                roi_dets,
                offset=(x1, y1),
                roi_size=(crop_w, crop_h),
                infer_size=(ROBOT_ROI_INFER_SIZE_PX, ROBOT_ROI_INFER_SIZE_PX),
                class_id=CLASS_ROBOT,
            )
            roi_dets = roi_dets[roi_dets.confidence > CONF_THRESHOLD] if len(roi_dets) > 0 else roi_dets
            roi_dets = self._filter_robot_roi_candidates(roi_dets, cached_box)
            if len(roi_dets) > 0:
                roi_detections.append(roi_dets)

        return self._combine_detections(roi_detections)

    @staticmethod
    def _robot_already_detected(cached_box: np.ndarray, fresh: sv.Detections) -> bool:
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

    @staticmethod
    def _filter_robot_roi_candidates(det: sv.Detections, cached_box: np.ndarray) -> sv.Detections:
        if len(det) == 0:
            return det

        cached_area = max(1.0, (cached_box[2] - cached_box[0]) * (cached_box[3] - cached_box[1]))
        cached_center = cv_utils.center(cached_box)
        keep = []
        for i in range(len(det)):
            box = det.xyxy[i]
            area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
            ratio = area / cached_area
            center = cv_utils.center(box)
            distance = np.hypot(center[0] - cached_center[0], center[1] - cached_center[1])
            if 0.35 <= ratio <= 2.8 and distance <= ROBOT_ROI_SIZE_PX * 0.45:
                keep.append(i)

        if not keep:
            return sv.Detections.empty()

        kept = det[np.array(keep)]
        best = int(np.argmax(kept.confidence)) if len(kept) > 1 else 0
        return kept[np.array([best])]

    @staticmethod
    def _scale_roi_detections_to_frame(
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

        # El predictor de ROI usa prompts solo de balon. Normalizamos todos los
        # class_id a CLASS_BALL para que nunca entren como robots.
        return sv.Detections(
            xyxy=xyxy,
            confidence=det.confidence,
            class_id=np.full(len(det), class_id, dtype=int),
        )

    @staticmethod
    def _offset_roi_detections_to_frame(
        det: sv.Detections,
        *,
        offset: tuple[int, int],
        class_id: int = CLASS_BALL,
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
        )

    @staticmethod
    def _roi_bounds(
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

    def process_frame(self, frame: cv2.typing.MatLike, frame_id: int) -> FrameResult:
        # 1. Inferencia base (full image)
        detections = self._infer_full_frame(frame)

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

        if len(robots_fresh) < MAX_ROBOTS and self._robot_cache is not None and self._robot_age <= MAX_ROBOT_AGE:
            robot_roi_fresh = self._infer_missing_robot_rois(frame, robots_fresh)
            robots_fresh = self._combine_detections([robots_fresh, robot_roi_fresh])
            robots_fresh = cv_utils.nms_detections(robots_fresh, 0.35)
            if len(robots_fresh) > MAX_ROBOTS:
                top = np.argsort(-robots_fresh.confidence)[:MAX_ROBOTS]
                robots_fresh = robots_fresh[top]

        # Persistencia por clase
        robots_fresh = self.robot_tracker.update(robots_fresh) if len(robots_fresh) > 0 else sv.Detections.empty()
        robots_det = self._get_robots(robots_fresh)
        robot_boxes = robots_det.xyxy if len(robots_det) > 0 else np.empty((0, 4), dtype=float)
        ball_candidates = self._get_ball_candidates(ball_fresh, frame, robot_boxes)

        # Si la imagen completa no da un candidato fisicamente valido, buscar
        # inmediatamente en un crop 256x256 centrado en la ultima posicion real.
        # Esto evita esperar un frame extra por _ball_age y recupera pelotas
        # pequenas que SAM pierde al escalar la imagen completa.
        if len(ball_candidates) == 0 and self._last_ball_center is not None:
            roi_ball_fresh, roi_bounds = self._infer_ball_roi(frame)
            if len(roi_ball_fresh) > 0:
                roi_ball_fresh = roi_ball_fresh[roi_ball_fresh.confidence > CONF_THRESHOLD]
            roi_candidates = self._get_ball_candidates(
                roi_ball_fresh,
                frame,
                robot_boxes,
                include_aux_sources=False,
            )
            roi_debug = dict(self.last_ball_debug)
            roi_debug.update({
                "roi_used": True,
                "roi_bounds": roi_bounds,
                "roi_raw_candidates": len(roi_ball_fresh),
            })
            self.last_ball_debug = roi_debug
            ball_candidates = roi_candidates

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
