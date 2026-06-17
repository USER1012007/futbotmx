from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import Iterable, Optional

import cv2
import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


from analysis.event_detector import EventDetector
from analysis.stats_engine import StatsEngine
from domain.entities import FrameResult, Point2D
from infra.configs import Config
from io_utils.tracking_io import TrackingIO
from visualization.dashboard import render_dashboard
from visualization.layout import compose_final_frame
from visualization.tactical_map import FieldStyle, TacticalMapRenderer
from visualization.video_render import render_video_overlay


FPS = 30
OUTPUT_SIZE = (1600, 900)
LEFT_WIDTH_RATIO = 0.64
VIDEO_PREVIEW_HEIGHT_RATIO = 0.50
TRACKING_PATH = Config.TRACKING_DIR / "tracking.jsonl"
OUTPUT_DIR = Config.OUTPUT_DIR / "mock_tracking_visualization"
FRAME_FIRST_PATH = OUTPUT_DIR / "frame_first.png"
FRAME_EVENT_PATH = OUTPUT_DIR / "frame_first_event.png"
FRAME_LAST_PATH = OUTPUT_DIR / "frame_last.png"
VIDEO_PATH = OUTPUT_DIR / "mock_tracking_visualization.mp4"
MAX_FRAMES: Optional[int] = None


def main() -> None:
    frames = TrackingIO(TRACKING_PATH).read_frame_results()
    if not frames:
        raise RuntimeError(f"No frames found in {TRACKING_PATH}")

    if MAX_FRAMES is not None:
        frames = frames[:MAX_FRAMES]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_w, output_h = OUTPUT_SIZE
    left_w = int(round(output_w * LEFT_WIDTH_RATIO))
    dashboard_w = output_w - left_w
    video_h = int(round(output_h * VIDEO_PREVIEW_HEIGHT_RATIO))
    tactical_h = output_h - video_h
    video_size = (left_w, video_h)

    pixel_bounds = _pixel_bounds(frames)
    event_detector = EventDetector(default_fps=FPS)
    stats_engine = StatsEngine()
    tactical_renderer = TacticalMapRenderer(
        style=FieldStyle(output_size=(left_w, tactical_h)),
        trail_length=60,
    )

    writer = cv2.VideoWriter(
        str(VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        OUTPUT_SIZE,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {VIDEO_PATH}")

    first_event_frame: Optional[np.ndarray] = None
    first_frame: Optional[np.ndarray] = None
    last_frame: Optional[np.ndarray] = None
    event_counts: dict[str, int] = {}

    try:
        for source_frame in frames:
            frame = _scale_frame_pixels(source_frame, pixel_bounds, video_size)
            frame_events = event_detector.detect(frame)
            stats = stats_engine.update(frame, frame_events)
            match_time_seconds = frame.timestamp_s if frame.timestamp_s is not None else frame.frame_id / FPS

            for event in frame_events.eventos:
                event_counts[event.type] = event_counts.get(event.type, 0) + 1

            raw_video = _mock_video_frame(frame.frame_id, video_size)
            video_overlay = render_video_overlay(raw_video, frame, frame_events)
            tactical_map = tactical_renderer.render(frame, frame_events)
            dashboard = render_dashboard(stats, frame_events, match_time_seconds, dashboard_w, output_h)

            final_frame = compose_final_frame(
                video_overlay,
                tactical_map,
                dashboard,
                output_size=OUTPUT_SIZE,
                left_width_ratio=LEFT_WIDTH_RATIO,
                video_preview_height_ratio=VIDEO_PREVIEW_HEIGHT_RATIO,
            )

            if first_frame is None:
                first_frame = final_frame
            if frame_events.eventos and first_event_frame is None:
                first_event_frame = final_frame

            writer.write(final_frame)
            last_frame = final_frame
    finally:
        writer.release()

    if first_frame is not None:
        cv2.imwrite(str(FRAME_FIRST_PATH), first_frame)
    if first_event_frame is not None:
        cv2.imwrite(str(FRAME_EVENT_PATH), first_event_frame)
    if last_frame is not None:
        cv2.imwrite(str(FRAME_LAST_PATH), last_frame)

    print(f"OK first frame: {FRAME_FIRST_PATH}")
    print(f"OK first event frame: {FRAME_EVENT_PATH if first_event_frame is not None else 'none'}")
    print(f"OK last frame: {FRAME_LAST_PATH}")
    print(f"OK video: {VIDEO_PATH}")
    print(f"frames: {len(frames)}")
    print(f"event_counts: {event_counts}")


def _pixel_bounds(frames: Iterable[FrameResult]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for frame in frames:
        for robot in frame.robots:
            xs.append(float(robot.position_pixel.x))
            ys.append(float(robot.position_pixel.y))
        if frame.ball is not None:
            xs.append(float(frame.ball.position_pixel.x))
            ys.append(float(frame.ball.position_pixel.y))

    if not xs or not ys:
        return 0.0, 1.0, 0.0, 1.0
    return min(xs), max(xs), min(ys), max(ys)


def _scale_frame_pixels(
    frame: FrameResult,
    bounds: tuple[float, float, float, float],
    video_size: tuple[int, int],
) -> FrameResult:
    robots = [
        replace(robot, position_pixel=_scale_point(robot.position_pixel, bounds, video_size))
        for robot in frame.robots
    ]
    ball = None
    if frame.ball is not None:
        ball = replace(frame.ball, position_pixel=_scale_point(frame.ball.position_pixel, bounds, video_size))

    return replace(
        frame,
        robots=robots,
        ball=ball,
        repositions=list(frame.repositions),
    )


def _scale_point(
    point: Point2D,
    bounds: tuple[float, float, float, float],
    video_size: tuple[int, int],
) -> Point2D:
    min_x, max_x, min_y, max_y = bounds
    width, height = video_size
    pad_x = max(24, int(width * 0.05))
    pad_y = max(18, int(height * 0.06))
    usable_w = max(1, width - 2 * pad_x)
    usable_h = max(1, height - 2 * pad_y)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    x = pad_x + (float(point.x) - min_x) / span_x * usable_w
    y = pad_y + (float(point.y) - min_y) / span_y * usable_h
    return Point2D(x, y, is_metric=False)


def _mock_video_frame(frame_id: int, video_size: tuple[int, int]) -> np.ndarray:
    width, height = video_size
    frame = np.full((height, width, 3), (28, 34, 42), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(30, 72, width, dtype=np.uint8)
    frame[:, :, 1] = np.linspace(38, 86, height, dtype=np.uint8)[:, None]
    cv2.rectangle(frame, (18, 18), (width - 18, height - 18), (72, 110, 82), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Tracking JSONL frame {frame_id}",
        (28, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (235, 238, 242),
        1,
        cv2.LINE_AA,
    )
    return frame


if __name__ == "__main__":
    main()
