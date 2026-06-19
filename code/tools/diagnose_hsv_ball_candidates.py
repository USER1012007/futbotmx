from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

cv2 = None
np = None
ball_utils = None


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


from infra.configs import Config


DEFAULT_FRAME_IDS = "0,8,60,63,74,90,120,150,181,234,240,253"


@dataclass(frozen=True)
class HsvBallCandidate:
    xyxy: np.ndarray
    center: tuple[float, float]
    area_px: float
    score: float
    source: str
    orange_pixels: float
    orange_ratio: float
    circularity: float
    s_field: float
    reject_reason: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose HSV ball candidates on selected video frames.")
    parser.add_argument("--video", type=Path, default=Config.VIDEO_DIR / "video1.mp4", help="Source video path.")
    parser.add_argument("--output-dir", type=Path, default=Config.OUTPUT_DIR / "ball_tracking_diagnostics", help="Output directory.")
    parser.add_argument("--frames", default=DEFAULT_FRAME_IDS, help="Comma-separated frame IDs.")
    parser.add_argument("--top-n", type=int, default=5, help="Candidates to draw per frame.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_cv_modules()
    _load_ball_helpers()
    frame_ids = [int(part.strip()) for part in args.frames.split(",") if part.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {args.video}")

    panels: list[np.ndarray] = []
    reject_counts: Counter[str] = Counter()
    try:
        for frame_id in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"{frame_id}: none")
                continue

            best, candidates = _score_hsv_candidates(frame)
            reject_counts.update(c.reject_reason or "accepted" for c in candidates)
            _draw_candidates(frame, candidates, best, args.top_n)
            if best is not None:
                cx, cy = best.center
                print(f"{frame_id}: ({cx:.1f}, {cy:.1f}) score={best.score:.3f}")
                _draw_crosshair(frame, (int(round(cx)), int(round(cy))))
                cv2.putText(
                    frame,
                    f"{best.source} {best.score:.2f}",
                    (int(round(cx)) + 14, int(round(cy)) - 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 149, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                print(f"{frame_id}: none ({len(candidates)} candidates)")

            cv2.putText(frame, f"Frame {frame_id}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(_resize_for_sheet(frame))
    finally:
        cap.release()

    if panels:
        _write_sheet(panels, args.output_dir / "hsv_ball_candidates_sheet.png", columns=3)
    print(f"reject_counts: {dict(reject_counts)}")


def _load_cv_modules() -> None:
    global cv2, np
    if cv2 is not None and np is not None:
        return
    import cv2 as cv2_module
    import numpy as np_module

    cv2 = cv2_module
    np = np_module


def _load_ball_helpers() -> None:
    global ball_utils
    if ball_utils is not None:
        return
    from vision import ball_utils as ball_utils_module

    ball_utils = ball_utils_module


def _score_hsv_candidates(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 100, 80], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8))
    mask2 = cv2.inRange(hsv, np.array([170, 100, 80], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    field_mask = _build_green_field_mask(frame)
    candidates = [_score_contour(frame, field_mask, contour) for contour in contours]
    accepted = [candidate for candidate in candidates if candidate.reject_reason is None]
    best = max(accepted, key=lambda candidate: candidate.score, default=None)
    return best, candidates


def _score_contour(frame: np.ndarray, field_mask: np.ndarray, contour) -> HsvBallCandidate:
    x, y, w, h = cv2.boundingRect(contour)
    xyxy = np.array([x, y, x + w, y + h], dtype=float)
    area = float(cv2.contourArea(contour))
    aspect = max(w / max(h, 1), h / max(w, 1))
    center = (x + w / 2.0, y + h / 2.0)

    signature = ball_utils.orange_signature_for_box(xyxy, frame)
    contour_circularity = _contour_circularity(contour)
    signature_circularity = float(signature["circularity"])
    circularity = max(contour_circularity, signature_circularity)
    field_ratio = ball_utils.field_context_ratio_for_box(xyxy, field_mask, frame.shape)
    center_on_field = ball_utils.center_on_field_for_box(xyxy, field_mask, frame.shape)
    orange_pixels = float(signature["orange_pixels"])
    orange_ratio = float(signature["orange_ratio"])

    reject_reason = None
    if area < ball_utils.BALL_MIN_AREA or area > ball_utils.BALL_MAX_AREA:
        reject_reason = "area"
    elif aspect > ball_utils.BALL_MAX_ASPECT_RATIO:
        reject_reason = "aspect"
    elif contour_circularity < ball_utils.BALL_MIN_CIRCULARITY:
        reject_reason = "circularity"
    elif (
        orange_pixels < ball_utils.BALL_MIN_ORANGE_PIXELS
        or orange_ratio < ball_utils.BALL_MIN_SIGNATURE_RATIO
        or signature_circularity < ball_utils.BALL_MIN_SIGNATURE_CIRCULARITY
    ):
        reject_reason = "orange_signature"
    elif not center_on_field and field_ratio < ball_utils.BALL_MIN_FIELD_NEIGHBOR_RATIO:
        reject_reason = "field_context"

    orange_score = min(1.0, orange_ratio / max(ball_utils.BALL_MIN_SIGNATURE_RATIO, 1e-6))
    field_score = min(1.0, field_ratio / max(ball_utils.BALL_MIN_FIELD_NEIGHBOR_RATIO, 1e-6))
    area_score = min(1.0, np.log1p(area) / np.log1p(ball_utils.BALL_MAX_AREA))
    score = 0.45 * circularity + 0.35 * orange_score + 0.15 * field_score + 0.05 * area_score
    if reject_reason is not None:
        score *= 0.25

    return HsvBallCandidate(
        xyxy=xyxy,
        center=center,
        area_px=area,
        score=float(score),
        source="hsv",
        orange_pixels=orange_pixels,
        orange_ratio=orange_ratio,
        circularity=float(circularity),
        s_field=float(field_ratio),
        reject_reason=reject_reason,
    )


def _contour_circularity(contour) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def _build_green_field_mask(frame: np.ndarray):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([35, 35, 30], dtype=np.uint8), np.array([95, 255, 255], dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    return (mask > 0).astype(np.uint8)


def _draw_candidates(frame: np.ndarray, candidates: list, best, top_n: int) -> None:
    ranked = sorted(
        candidates,
        key=lambda c: (c.reject_reason is None, c.score, c.area_px),
        reverse=True,
    )[: max(1, top_n)]
    for index, candidate in enumerate(ranked, start=1):
        x0, y0, x1, y1 = [int(round(v)) for v in candidate.xyxy]
        if best is not None and candidate is best:
            color = (0, 149, 255)
        elif candidate.reject_reason is None:
            color = (80, 220, 80)
        else:
            color = (150, 150, 150)
        cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        label = (
            f"{index} s={candidate.score:.2f} f={candidate.s_field:.2f} "
            f"o={candidate.orange_ratio:.2f} c={candidate.circularity:.2f}"
        )
        if candidate.reject_reason:
            label += f" {candidate.reject_reason}"
        label_y = max(18, y0 - 8)
        cv2.putText(frame, label, (max(4, x0), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_crosshair(frame: np.ndarray, point: tuple[int, int]) -> None:
    x, y = point
    color = (0, 149, 255)
    cv2.circle(frame, point, 12, color, 2, cv2.LINE_AA)
    cv2.line(frame, (x - 28, y), (x - 14, y), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x + 14, y), (x + 28, y), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y - 28), (x, y - 14), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y + 14), (x, y + 28), color, 2, cv2.LINE_AA)


def _resize_for_sheet(frame: np.ndarray) -> np.ndarray:
    rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(rotated, (480, 854), interpolation=cv2.INTER_AREA)


def _write_sheet(panels: list[np.ndarray], path: Path, columns: int) -> None:
    rows = int(np.ceil(len(panels) / columns))
    h, w = panels[0].shape[:2]
    sheet = np.full((rows * h, columns * w, 3), (20, 20, 20), dtype=np.uint8)
    for index, panel in enumerate(panels):
        row = index // columns
        col = index % columns
        sheet[row * h : row * h + h, col * w : col * w + w] = panel
    cv2.imwrite(str(path), sheet)


if __name__ == "__main__":
    main()
