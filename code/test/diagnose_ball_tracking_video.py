from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


from infra.configs import Config


SOURCE_VIDEO_PATH = Config.VIDEO_DIR / "video1.mp4"
TRACKING_PATH = Config.TRACKING_DIR / "tracking.jsonl"
OUTPUT_DIR = Config.OUTPUT_DIR / "ball_tracking_diagnostics"
VIDEO_PANEL_SIZE = (1024, 450)
ROTATION = "clockwise"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = _read_tracking()
    frame_ids = _pick_frame_ids(records)
    panels = [_render_panel(frame_id, records[frame_id]) for frame_id in frame_ids if frame_id in records]
    _write_sheet(panels, OUTPUT_DIR / "ball_detection_sheet.png", columns=2)
    _write_jump_report(records, OUTPUT_DIR / "ball_jump_report.txt")
    print(f"OK sheet: {OUTPUT_DIR / 'ball_detection_sheet.png'}")
    print(f"OK report: {OUTPUT_DIR / 'ball_jump_report.txt'}")
    print(f"frames: {frame_ids}")


def _read_tracking() -> dict[int, dict]:
    records: dict[int, dict] = {}
    for line in TRACKING_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        data = raw.get("data", raw)
        records[int(data["frame_id"])] = data
    return records


def _pick_frame_ids(records: dict[int, dict]) -> list[int]:
    # Actual frames in video: 254
    max_frame = 250 
    ordered = [f for f in sorted(records) if f <= max_frame]
    sampled = set(ordered[::20])
    sampled.update({0, 1, 2, 32, 60, 90, 120, 150, 180, 210, 240})

    jumps: list[tuple[float, int]] = []
    previous = None
    for frame_id in ordered:
        ball = records[frame_id].get("ball")
        if not ball:
            previous = None
            continue
        pos = ball["position_pixel"]
        point = (float(pos["x"]), float(pos["y"]))
        if previous is not None:
            previous_frame_id, previous_point = previous
            frame_delta = max(frame_id - previous_frame_id, 1)
            dist = math.hypot(point[0] - previous_point[0], point[1] - previous_point[1]) / frame_delta
            jumps.append((dist, frame_id))
        previous = (frame_id, point)

    for _, frame_id in sorted(jumps, reverse=True)[:12]:
        sampled.update({max(0, frame_id - 1), frame_id, frame_id + 1})

    return [frame_id for frame_id in sorted(sampled) if frame_id in records]


def _render_panel(frame_id: int, record: dict) -> np.ndarray:
    frame, transform = _read_video_frame(frame_id)
    ball = record.get("ball")
    robots = record.get("robots", [])

    if ball:
        ball_point = _transform_point(ball["position_pixel"], transform)
        _draw_crosshair(frame, ball_point, (0, 149, 255))
        cv2.putText(frame, "tracked", (ball_point[0] + 14, ball_point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    for robot in robots:
        point = _transform_point(robot["position_pixel"], transform)
        cv2.circle(frame, point, 16, (60, 35, 239) if robot.get("team_id") == "rivals" else (216, 180, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, robot.get("id", "R"), (point[0] + 18, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, f"Frame {frame_id}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    if not ball:
        return frame

    crop = _crop_around(frame, ball_point, 180)
    crop = cv2.resize(crop, (360, 360), interpolation=cv2.INTER_NEAREST)
    panel = np.full((max(frame.shape[0], crop.shape[0]), frame.shape[1] + crop.shape[1], 3), (20, 20, 20), dtype=np.uint8)
    panel[: frame.shape[0], : frame.shape[1]] = frame
    panel[: crop.shape[0], frame.shape[1] :] = crop
    return panel


def _draw_crosshair(frame: np.ndarray, point: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = point
    cv2.circle(frame, point, 8, color, 2, cv2.LINE_AA)
    cv2.line(frame, (x - 18, y), (x - 10, y), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x + 10, y), (x + 18, y), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y - 18), (x, y - 10), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y + 10), (x, y + 18), color, 2, cv2.LINE_AA)


def _read_video_frame(frame_id: int) -> tuple[np.ndarray, dict]:
    cap = cv2.VideoCapture(str(SOURCE_VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {SOURCE_VIDEO_PATH}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_id}")

    source_h, source_w = frame.shape[:2]
    if ROTATION == "clockwise":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif ROTATION == "counterclockwise":
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    target_w, target_h = VIDEO_PANEL_SIZE
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))
    offset_x = (target_w - resized_w) // 2
    offset_y = (target_h - resized_h) // 2
    rendered = np.full((target_h, target_w, 3), (14, 18, 24), dtype=np.uint8)
    rendered[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    return rendered, {
        "source_w": source_w,
        "source_h": source_h,
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }


def _transform_point(point: dict, transform: dict) -> tuple[int, int]:
    x = float(point["x"])
    y = float(point["y"])
    if ROTATION == "clockwise":
        x, y = transform["source_h"] - 1.0 - y, x
    elif ROTATION == "counterclockwise":
        x, y = y, transform["source_w"] - 1.0 - x
    x = transform["offset_x"] + x * transform["scale"]
    y = transform["offset_y"] + y * transform["scale"]
    return int(round(x)), int(round(y))


def _crop_around(frame: np.ndarray, center: tuple[int, int], size: int) -> np.ndarray:
    half = size // 2
    x, y = center
    padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    x += half
    y += half
    return padded[y - half : y + half, x - half : x + half]


def _write_sheet(panels: list[np.ndarray], path: Path, columns: int) -> None:
    if not panels:
        return
    width = max(panel.shape[1] for panel in panels)
    height = max(panel.shape[0] for panel in panels)
    rows = int(math.ceil(len(panels) / columns))
    sheet = np.full((rows * height, columns * width, 3), (20, 20, 20), dtype=np.uint8)
    for index, panel in enumerate(panels):
        row = index // columns
        col = index % columns
        sheet[row * height : row * height + panel.shape[0], col * width : col * width + panel.shape[1]] = panel
    cv2.imwrite(str(path), sheet)


def _write_jump_report(records: dict[int, dict], path: Path) -> None:
    lines = ["frame_id,raw_pixel_x,raw_pixel_y,delta_px_per_frame"]
    previous = None
    for frame_id in sorted(records):
        ball = records[frame_id].get("ball")
        if not ball:
            previous = None
            continue
        pos = ball["position_pixel"]
        point = (float(pos["x"]), float(pos["y"]))
        delta = 0.0
        if previous is not None:
            previous_frame_id, previous_point = previous
            frame_delta = max(frame_id - previous_frame_id, 1)
            delta = math.hypot(point[0] - previous_point[0], point[1] - previous_point[1]) / frame_delta
        lines.append(f"{frame_id},{point[0]:.1f},{point[1]:.1f},{delta:.1f}")
        previous = (frame_id, point)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
