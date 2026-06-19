import cv2
import numpy as np
import supervision as sv

# ── Constantes ────────────────────────────────────────────────────────────────
BALL_MIN_AREA              = 8          # más pequeño: bola puede estar lejos
BALL_MAX_AREA              = 3500
BALL_MAX_ASPECT_RATIO      = 1.8
BALL_MIN_CIRCULARITY       = 0.45       # relajado: bola parcialmente ocluida
BALL_MIN_ORANGE_PIXELS     = 6
BALL_MIN_ORANGE_RATIO      = 0.04
BALL_MIN_SIGNATURE_RATIO   = 0.12
BALL_MIN_SIGNATURE_CIRCULARITY = 0.58
BALL_MIN_FIELD_NEIGHBOR_RATIO = 0.18
BALL_HSV_CONFIDENCE        = 0.92
BALL_ROBOT_REJECT_RADIUS_PX   = 58.0
BALL_ROBOT_REJECT_PADDING_PX  = 14.0
BALL_FIELD_CONTEXT_PADDING_PX = 18
BALL_SKIN_CONTEXT_PADDING_PX = 42
BALL_MAX_SKIN_RING_RATIO = 0.14
BALL_MAX_SKIN_BOX_RATIO = 0.28
BALL_MIN_SKIN_PIXELS_TO_REJECT = 90

# ── Rangos HSV naranja (dos bandas para capturar sombras/sobreexposición) ─────
# Banda primaria: naranja puro brillante
_HSV_ORANGE_LO1 = np.array([0,  120,  80], dtype=np.uint8)
_HSV_ORANGE_HI1 = np.array([10, 255, 255], dtype=np.uint8)
# Banda secundaria: naranja oscuro / bajo brillo (bola en sombra del robot)
_HSV_ORANGE_LO2 = np.array([165, 100,  60], dtype=np.uint8)
_HSV_ORANGE_HI2 = np.array([180, 255, 255], dtype=np.uint8)
# Banda terciaria: naranja-amarillo (sobreexposición, luz fuerte encima)
_HSV_ORANGE_LO3 = np.array([10,  115,  110], dtype=np.uint8)
_HSV_ORANGE_HI3 = np.array([25, 255,  255], dtype=np.uint8)


# ── Helpers internos ──────────────────────────────────────────────────────────
def _box_area(xyxy: np.ndarray) -> float:
    return max(0, xyxy[2] - xyxy[0]) * max(0, xyxy[3] - xyxy[1])


def _center(xyxy: np.ndarray):
    return ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)


def _orange_mask(frame: np.ndarray) -> np.ndarray:
    """Máscara HSV triple-banda para naranja robusto."""
    if frame is None or frame.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, _HSV_ORANGE_LO1, _HSV_ORANGE_HI1)
    m2 = cv2.inRange(hsv, _HSV_ORANGE_LO2, _HSV_ORANGE_HI2)
    m3 = cv2.inRange(hsv, _HSV_ORANGE_LO3, _HSV_ORANGE_HI3)
    return cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)


def _skin_mask(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin_ycrcb = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin_hsv_1 = cv2.inRange(
        hsv,
        np.array([0, 20, 70], dtype=np.uint8),
        np.array([25, 170, 255], dtype=np.uint8),
    )
    skin_hsv_2 = cv2.inRange(
        hsv,
        np.array([165, 20, 70], dtype=np.uint8),
        np.array([180, 170, 255], dtype=np.uint8),
    )
    return cv2.bitwise_or(skin_ycrcb, cv2.bitwise_or(skin_hsv_1, skin_hsv_2))


def _clip_box(xyxy: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(width,  int(np.floor(xyxy[0]))))
    y0 = max(0, min(height, int(np.floor(xyxy[1]))))
    x1 = max(0, min(width,  int(np.ceil(xyxy[2]))))
    y1 = max(0, min(height, int(np.ceil(xyxy[3]))))
    return x0, y0, x1, y1


def _is_near_robot(center: tuple[float, float], robot_boxes: np.ndarray) -> bool:
    cx, cy = center
    for box in robot_boxes:
        x0, y0, x1, y1 = box
        exp_x0 = x0 - BALL_ROBOT_REJECT_PADDING_PX
        exp_y0 = y0 - BALL_ROBOT_REJECT_PADDING_PX
        exp_x1 = x1 + BALL_ROBOT_REJECT_PADDING_PX
        exp_y1 = y1 + BALL_ROBOT_REJECT_PADDING_PX
        if exp_x0 <= cx <= exp_x1 and exp_y0 <= cy <= exp_y1:
            return True
        rx, ry = _center(box)
        if np.hypot(cx - rx, cy - ry) < BALL_ROBOT_REJECT_RADIUS_PX:
            return True
    return False


def _circularity(cnt) -> float:
    area  = cv2.contourArea(cnt)
    perim = cv2.arcLength(cnt, True)
    if perim == 0:
        return 0.0
    return 4 * np.pi * area / (perim ** 2)


# ── Filtros de detecciones sv.Detections ─────────────────────────────────────
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
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        if max(w / h, h / w) <= BALL_MAX_ASPECT_RATIO:
            keep.append(i)
    return det[np.array(keep)] if keep else sv.Detections.empty()


def filter_ball_by_orange_support(det: sv.Detections, frame: np.ndarray) -> sv.Detections:
    if len(det) == 0:
        return det
    if frame is None or frame.size == 0:
        return sv.Detections.empty()
    orange_mask = _orange_mask(frame)
    keep = []
    for i in range(len(det)):
        x0, y0, x1, y1 = _clip_box(det.xyxy[i], frame.shape[1], frame.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        roi           = orange_mask[y0:y1, x0:x1]
        orange_pixels = int(np.count_nonzero(roi))
        orange_ratio  = orange_pixels / max(1, roi.size)
        if orange_pixels >= BALL_MIN_ORANGE_PIXELS and orange_ratio >= BALL_MIN_ORANGE_RATIO:
            keep.append(i)
    return det[np.array(keep)] if keep else sv.Detections.empty()


def filter_ball_by_orange_signature(det: sv.Detections, frame: np.ndarray) -> sv.Detections:
    if len(det) == 0:
        return det
    keep = []
    for i in range(len(det)):
        signature = orange_signature_for_box(det.xyxy[i], frame)
        if (
            signature["orange_pixels"] >= BALL_MIN_ORANGE_PIXELS
            and signature["orange_ratio"] >= BALL_MIN_SIGNATURE_RATIO
            and signature["circularity"] >= BALL_MIN_SIGNATURE_CIRCULARITY
        ):
            keep.append(i)
    return det[np.array(keep)] if keep else sv.Detections.empty()


def filter_ball_against_skin(det: sv.Detections, frame: np.ndarray) -> sv.Detections:
    if len(det) == 0:
        return det
    keep = [
        i
        for i in range(len(det))
        if not is_skin_like_false_ball(det.xyxy[i], frame)
    ]
    return det[np.array(keep)] if keep else sv.Detections.empty()


def filter_ball_against_robots(det: sv.Detections, robot_boxes: np.ndarray) -> sv.Detections:
    if len(det) == 0 or len(robot_boxes) == 0:
        return det
    keep = [i for i in range(len(det)) if not _is_near_robot(_center(det.xyxy[i]), robot_boxes)]
    return det[np.array(keep)] if keep else sv.Detections.empty()


def filter_ball_by_field_context(
    det: sv.Detections,
    field_mask: np.ndarray | None,
    frame_shape: tuple[int, ...],
) -> sv.Detections:
    if len(det) == 0 or field_mask is None:
        return det

    field = _normalized_field_mask(field_mask, frame_shape[1], frame_shape[0])
    kernel = np.ones((15, 15), np.uint8)
    dilated = cv2.dilate(field, kernel, iterations=2)

    keep = []
    for i in range(len(det)):
        cx, cy = _center(det.xyxy[i])
        px = int(round(cx))
        py = int(round(cy))
        center_on_field = 0 <= px < dilated.shape[1] and 0 <= py < dilated.shape[0] and dilated[py, px] > 0

        x0, y0, x1, y1 = _expanded_box(
            det.xyxy[i],
            frame_shape[1],
            frame_shape[0],
            BALL_FIELD_CONTEXT_PADDING_PX,
        )
        roi = field[y0:y1, x0:x1]
        neighbor_ratio = int(np.count_nonzero(roi)) / max(1, roi.size)
        if center_on_field or neighbor_ratio >= BALL_MIN_FIELD_NEIGHBOR_RATIO:
            keep.append(i)

    return det[np.array(keep)] if keep else sv.Detections.empty()


def filter_ball_center_on_field(
    det: sv.Detections,
    field_mask: np.ndarray | None,
    frame_shape: tuple[int, ...],
) -> sv.Detections:
    if len(det) == 0 or field_mask is None:
        return det

    keep = [
        i
        for i in range(len(det))
        if center_on_field_for_box(det.xyxy[i], field_mask, frame_shape)
    ]
    return det[np.array(keep)] if keep else sv.Detections.empty()


def orange_ratio_for_box(xyxy: np.ndarray, frame: np.ndarray) -> float:
    """Devuelve fracción de píxeles naranja dentro del bounding box."""
    if frame is None or frame.size == 0:
        return 0.0
    orange_mask = _orange_mask(frame)
    x0, y0, x1, y1 = _clip_box(xyxy, frame.shape[1], frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = orange_mask[y0:y1, x0:x1]
    return int(np.count_nonzero(roi)) / max(1, roi.size)


def orange_signature_for_box(xyxy: np.ndarray, frame: np.ndarray) -> dict[str, float]:
    if frame is None or frame.size == 0:
        return {"orange_pixels": 0.0, "orange_ratio": 0.0, "circularity": 0.0}
    orange_mask = _orange_mask(frame)
    x0, y0, x1, y1 = _clip_box(xyxy, frame.shape[1], frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return {"orange_pixels": 0.0, "orange_ratio": 0.0, "circularity": 0.0}

    roi = orange_mask[y0:y1, x0:x1]
    orange_pixels = int(np.count_nonzero(roi))
    orange_ratio = orange_pixels / max(1, roi.size)

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    if contours:
        circularity = max(_circularity(cnt) for cnt in contours)

    return {
        "orange_pixels": float(orange_pixels),
        "orange_ratio": float(orange_ratio),
        "circularity": float(circularity),
    }


def skin_context_for_box(xyxy: np.ndarray, frame: np.ndarray) -> dict[str, float]:
    if frame is None or frame.size == 0:
        return {"skin_ring_pixels": 0.0, "skin_ring_ratio": 0.0, "skin_box_ratio": 0.0}

    height, width = frame.shape[:2]
    skin = _skin_mask(frame)

    x0, y0, x1, y1 = _clip_box(xyxy, width, height)
    if x1 <= x0 or y1 <= y0:
        return {"skin_ring_pixels": 0.0, "skin_ring_ratio": 0.0, "skin_box_ratio": 0.0}

    ex0, ey0, ex1, ey1 = _expanded_box(
        xyxy,
        width,
        height,
        BALL_SKIN_CONTEXT_PADDING_PX,
    )
    expanded = skin[ey0:ey1, ex0:ex1]
    box_skin = skin[y0:y1, x0:x1]
    ring_mask = np.ones(expanded.shape, dtype=np.uint8)
    local_x0 = x0 - ex0
    local_y0 = y0 - ey0
    local_x1 = x1 - ex0
    local_y1 = y1 - ey0
    ring_mask[local_y0:local_y1, local_x0:local_x1] = 0
    skin_ring_pixels = int(np.count_nonzero(cv2.bitwise_and(expanded, expanded, mask=ring_mask)))
    ring_pixels = max(1, int(np.count_nonzero(ring_mask)))
    skin_box_pixels = int(np.count_nonzero(box_skin))

    return {
        "skin_ring_pixels": float(skin_ring_pixels),
        "skin_ring_ratio": float(skin_ring_pixels / ring_pixels),
        "skin_box_ratio": float(skin_box_pixels / max(1, box_skin.size)),
    }


def is_skin_like_false_ball(xyxy: np.ndarray, frame: np.ndarray) -> bool:
    context = skin_context_for_box(xyxy, frame)
    if (
        context["skin_ring_pixels"] >= BALL_MIN_SKIN_PIXELS_TO_REJECT
        and context["skin_ring_ratio"] >= BALL_MAX_SKIN_RING_RATIO
    ):
        return True
    if context["skin_box_ratio"] >= BALL_MAX_SKIN_BOX_RATIO:
        signature = orange_signature_for_box(xyxy, frame)
        return signature["orange_ratio"] < 0.35
    return False


def field_context_ratio_for_box(
    xyxy: np.ndarray,
    field_mask: np.ndarray | None,
    frame_shape: tuple[int, ...],
) -> float:
    if field_mask is None:
        return 1.0
    field = _normalized_field_mask(field_mask, frame_shape[1], frame_shape[0])
    x0, y0, x1, y1 = _expanded_box(
        xyxy,
        frame_shape[1],
        frame_shape[0],
        BALL_FIELD_CONTEXT_PADDING_PX,
    )
    roi = field[y0:y1, x0:x1]
    return int(np.count_nonzero(roi)) / max(1, roi.size)


def center_on_field_for_box(
    xyxy: np.ndarray,
    field_mask: np.ndarray | None,
    frame_shape: tuple[int, ...],
) -> bool:
    if field_mask is None:
        return True
    field = _normalized_field_mask(field_mask, frame_shape[1], frame_shape[0])
    cx, cy = _center(xyxy)
    px = int(round(cx))
    py = int(round(cy))
    return 0 <= px < field.shape[1] and 0 <= py < field.shape[0] and field[py, px] > 0


def _normalized_field_mask(field_mask: np.ndarray, width: int, height: int) -> np.ndarray:
    fm = field_mask
    if fm.ndim == 3:
        fm = np.any(fm, axis=0).astype(np.uint8)
    fm = (fm * 255).astype(np.uint8) if fm.max() <= 1 else fm.astype(np.uint8)
    fm = cv2.resize(fm, (width, height), interpolation=cv2.INTER_NEAREST)
    return (fm > 0).astype(np.uint8) * 255


def _expanded_box(
    xyxy: np.ndarray,
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int]:
    return _clip_box(
        np.array([
            xyxy[0] - padding,
            xyxy[1] - padding,
            xyxy[2] + padding,
            xyxy[3] + padding,
        ]),
        width,
        height,
    )


# ── Fallback HSV puro ─────────────────────────────────────────────────────────
def hsv_ball_fallback(
    frame: np.ndarray,
    field_mask: np.ndarray | None,
    robot_boxes: np.ndarray | None = None,
    max_candidates: int = 1,
) -> sv.Detections:
    if frame is None or frame.size == 0:
        return sv.Detections.empty()

    mask = _orange_mask(frame)
    if mask.size == 0:
        return sv.Detections.empty()

    if field_mask is not None:
        fm = _normalized_field_mask(field_mask, mask.shape[1], mask.shape[0])
        mask = cv2.bitwise_and(mask, fm)

    # Morfología más agresiva: cerrar huecos pequeños antes de buscar contornos
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  k3, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, k5, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return sv.Detections.empty()

    candidates: list[tuple[float, np.ndarray]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (BALL_MIN_AREA <= area <= BALL_MAX_AREA):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if max(w / max(h, 1), h / max(w, 1)) > BALL_MAX_ASPECT_RATIO:
            continue
        center = (x + w / 2.0, y + h / 2.0)
        if robot_boxes is not None and len(robot_boxes) > 0 and _is_near_robot(center, robot_boxes):
            continue
        circ = _circularity(cnt)
        if circ < BALL_MIN_CIRCULARITY:
            continue
        box = np.array([x, y, x + w, y + h], dtype=float)
        if is_skin_like_false_ball(box, frame):
            continue
        # Score combinado: circularity * log(area) para preferir pelotas grandes y redondas
        score = circ * np.log1p(area)
        candidates.append((float(score), box))

    if not candidates:
        return sv.Detections.empty()

    candidates.sort(key=lambda item: item[0], reverse=True)
    boxes = [box for _, box in candidates[:max(1, max_candidates)]]

    return sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        confidence=np.full(len(boxes), BALL_HSV_CONFIDENCE, dtype=float),
        class_id=np.full(len(boxes), 1, dtype=int),
        tracker_id=np.full(len(boxes), -1, dtype=int),
    )


# ── Template matching naranja ──────────────────────────────────────────────────
def template_match_ball(
    frame: np.ndarray,
    field_mask: np.ndarray | None,
    search_center: tuple[float, float] | None = None,
    search_radius: float = 120.0,
    robot_boxes: np.ndarray | None = None,
    template_sizes: list[int] | None = None,
) -> sv.Detections:
    """
    Crea templates sintéticos de círculo naranja y los busca en el frame.
    Útil cuando SAM y HSV fallan (oclusión parcial, iluminación extrema).
    """
    if template_sizes is None:
        template_sizes = [8, 12, 16, 20, 26]

    if frame is None or frame.size == 0:
        return sv.Detections.empty()

    # Región de búsqueda
    h, w = frame.shape[:2]
    if search_center is not None:
        cx, cy = search_center
        x0 = max(0, int(cx - search_radius))
        y0 = max(0, int(cy - search_radius))
        x1 = min(w, int(cx + search_radius))
        y1 = min(h, int(cy + search_radius))
        if x1 <= x0 or y1 <= y0:
            return sv.Detections.empty()
        roi_frame = frame[y0:y1, x0:x1]
        offset    = (x0, y0)
    else:
        roi_frame = frame
        offset    = (0, 0)

    if roi_frame.size == 0:
        return sv.Detections.empty()

    if field_mask is not None:
        fm = (field_mask * 255).astype(np.uint8) if field_mask.max() <= 1 else field_mask
        fm = cv2.resize(fm, (w, h), interpolation=cv2.INTER_NEAREST)
        roi_fm = fm[offset[1]:offset[1]+roi_frame.shape[0],
                    offset[0]:offset[0]+roi_frame.shape[1]]
    else:
        roi_fm = None

    # Canal naranja para matching
    orange_mask_roi = _orange_mask(roi_frame)
    if orange_mask_roi.size == 0:
        return sv.Detections.empty()
    if roi_fm is not None:
        orange_mask_roi = cv2.bitwise_and(orange_mask_roi, roi_fm)

    best_val  = -1.0
    best_box  = None

    for size in template_sizes:
        if size > min(roi_frame.shape[:2]):
            continue
        # Template: disco blanco (= naranja) sobre fondo negro
        tmpl = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(tmpl, (size // 2, size // 2), size // 2 - 1, 255, -1)

        res = cv2.matchTemplate(orange_mask_roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val > best_val:
            tx, ty = max_loc
            gx = tx + offset[0]
            gy = ty + offset[1]
            candidate_center = (gx + size / 2, gy + size / 2)
            if robot_boxes is not None and _is_near_robot(candidate_center, robot_boxes):
                continue
            best_val = max_val
            best_box = np.array([gx, gy, gx + size, gy + size], dtype=float)

    if best_box is None or best_val < 0.35:   # umbral de correlación mínima
        return sv.Detections.empty()

    conf = float(np.clip(best_val * 0.88, 0.0, 1.0))   # escalar a confianza
    return sv.Detections(
        xyxy=best_box[np.newaxis],
        confidence=np.array([conf]),
        class_id=np.array([1]),
        tracker_id=np.array([-1]),
    )
