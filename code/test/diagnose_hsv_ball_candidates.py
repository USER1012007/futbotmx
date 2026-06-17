from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


from infra.configs import Config
from vision.segmentation import _center, _hsv_ball_fallback


SOURCE_VIDEO_PATH = Config.VIDEO_DIR / "video1.mp4"
OUTPUT_DIR = Config.OUTPUT_DIR / "ball_tracking_diagnostics"
FRAME_IDS = [0, 8, 60, 63, 74, 90, 120, 150, 181, 234, 240, 253]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(SOURCE_VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {SOURCE_VIDEO_PATH}")

    panels: list[np.ndarray] = []
    try:
        for frame_id in FRAME_IDS:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"{frame_id}: none")
                continue

            det = _hsv_ball_fallback(frame, None, np.empty((0, 4), dtype=float))
            if len(det) > 0:
                cx, cy = _center(det.xyxy[0])
                print(f"{frame_id}: ({cx:.1f}, {cy:.1f})")
                _draw_crosshair(frame, (int(round(cx)), int(round(cy))))
            else:
                print(f"{frame_id}: none")

            cv2.putText(frame, f"Frame {frame_id}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(_resize_for_sheet(frame))
    finally:
        cap.release()

    if panels:
        _write_sheet(panels, OUTPUT_DIR / "hsv_ball_candidates_sheet.png", columns=3)


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
