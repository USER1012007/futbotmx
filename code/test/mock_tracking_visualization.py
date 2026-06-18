from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Literal, Optional

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
from visualization.dashboard import DashboardRenderer
from visualization.layout import compose_final_frame
from visualization.tactical_map import FieldStyle, TacticalMapRenderer
from visualization.video_render import render_video_overlay


FPS = 30
OUTPUT_SIZE = (1600, 900)
LEFT_WIDTH_RATIO = 0.64
VIDEO_PREVIEW_HEIGHT_RATIO = 0.50
TRACKING_PATH = Config.TRACKING_DIR / "tracking.jsonl"
SOURCE_VIDEO_PATH = Config.VIDEO_DIR / "video1.mp4"
# SOURCE_VIDEO_PATH = Config.VIDEO_DIR / "video2.MOV"
OUTPUT_DIR = Config.OUTPUT_DIR / "mock_tracking_visualization"
FRAME_FIRST_PATH = OUTPUT_DIR / "frame_first.png"
FRAME_EVENT_PATH = OUTPUT_DIR / "frame_first_event.png"
FRAME_LAST_PATH = OUTPUT_DIR / "frame_last.png"
VIDEO_PATH = OUTPUT_DIR / "mock_tracking_visualization.mp4"
MAX_FRAMES: Optional[int] = None
VIDEO_ROTATION: Literal["clockwise", "counterclockwise", "none"] = "clockwise"
LETTERBOX_COLOR = (14, 18, 24)


@dataclass(frozen=True)
class VideoFrameTransform:
    source_width: int
    source_height: int
    rotated_width: int
    rotated_height: int
    scale: float
    offset_x: int
    offset_y: int


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

    event_detector = EventDetector(default_fps=FPS)
    stats_engine = StatsEngine()
    dashboard_renderer = DashboardRenderer()
    tactical_renderer = TacticalMapRenderer(
        style=FieldStyle(output_size=(left_w, tactical_h), mirror_x=True),
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

    source_capture = cv2.VideoCapture(str(SOURCE_VIDEO_PATH))
    if not source_capture.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open source video {SOURCE_VIDEO_PATH}")

    first_event_frame: Optional[np.ndarray] = None
    first_frame: Optional[np.ndarray] = None
    last_frame: Optional[np.ndarray] = None
    event_counts: dict[str, int] = {}

    try:
        for source_frame in frames:
            raw_video, transform = _read_render_frame(
                source_capture,
                source_frame.frame_id,
                video_size,
                rotation=VIDEO_ROTATION,
            )
            frame = _transform_frame_pixels(source_frame, transform, VIDEO_ROTATION)
            frame_events = event_detector.detect(frame)
            stats = stats_engine.update(frame, frame_events)
            match_time_seconds = frame.timestamp_s if frame.timestamp_s is not None else frame.frame_id / FPS

            for event in frame_events.eventos:
                event_counts[event.type] = event_counts.get(event.type, 0) + 1

            video_overlay = render_video_overlay(raw_video, frame, frame_events)
            tactical_map = tactical_renderer.render(frame, frame_events)
            dashboard = dashboard_renderer.render(stats, frame_events, match_time_seconds, dashboard_w, output_h)

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
        source_capture.release()
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


def _read_render_frame(
    capture: cv2.VideoCapture,
    frame_id: int,
    output_size: tuple[int, int],
    *,
    rotation: Literal["clockwise", "counterclockwise", "none"],
) -> tuple[np.ndarray, VideoFrameTransform]:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_id} from {SOURCE_VIDEO_PATH}")

    source_h, source_w = frame.shape[:2]
    rotated = _rotate_frame(frame, rotation)
    rendered, transform = _letterbox_frame(
        rotated,
        output_size,
        source_width=source_w,
        source_height=source_h,
    )
    return rendered, transform


def _rotate_frame(
    frame: np.ndarray,
    rotation: Literal["clockwise", "counterclockwise", "none"],
) -> np.ndarray:
    if rotation == "clockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "counterclockwise":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "none":
        return frame
    raise ValueError(f"Unsupported video rotation: {rotation}")


def _letterbox_frame(
    frame: np.ndarray,
    output_size: tuple[int, int],
    *,
    source_width: int,
    source_height: int,
) -> tuple[np.ndarray, VideoFrameTransform]:
    output_w, output_h = output_size
    frame_h, frame_w = frame.shape[:2]
    scale = min(output_w / frame_w, output_h / frame_h)
    resized_w = max(1, int(round(frame_w * scale)))
    resized_h = max(1, int(round(frame_h * scale)))
    offset_x = (output_w - resized_w) // 2
    offset_y = (output_h - resized_h) // 2

    rendered = np.full((output_h, output_w, 3), LETTERBOX_COLOR, dtype=np.uint8)
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    rendered[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized

    transform = VideoFrameTransform(
        source_width=source_width,
        source_height=source_height,
        rotated_width=frame_w,
        rotated_height=frame_h,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    return rendered, transform


def _transform_frame_pixels(
    frame: FrameResult,
    transform: VideoFrameTransform,
    rotation: Literal["clockwise", "counterclockwise", "none"],
) -> FrameResult:
    robots = [
        replace(robot, position_pixel=_transform_point(robot.position_pixel, transform, rotation))
        for robot in frame.robots
    ]
    ball = None
    if frame.ball is not None:
        ball = replace(frame.ball, position_pixel=_transform_point(frame.ball.position_pixel, transform, rotation))

    return replace(
        frame,
        robots=robots,
        ball=ball,
        repositions=list(frame.repositions),
    )


def _transform_point(
    point: Point2D,
    transform: VideoFrameTransform,
    rotation: Literal["clockwise", "counterclockwise", "none"],
) -> Point2D:
    x, y = _rotate_point(float(point.x), float(point.y), transform, rotation)
    x = transform.offset_x + x * transform.scale
    y = transform.offset_y + y * transform.scale
    return Point2D(x, y, is_metric=False)


def _rotate_point(
    x: float,
    y: float,
    transform: VideoFrameTransform,
    rotation: Literal["clockwise", "counterclockwise", "none"],
) -> tuple[float, float]:
    if rotation == "clockwise":
        return transform.source_height - 1.0 - y, x
    if rotation == "counterclockwise":
        return y, transform.source_width - 1.0 - x
    if rotation == "none":
        return x, y
    raise ValueError(f"Unsupported video rotation: {rotation}")


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
