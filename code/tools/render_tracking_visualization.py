from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Literal, Optional

cv2 = None
np = None
DashboardRenderer = None
EventDetector = None
FieldStyle = None
FrameResult = None
Point2D = None
StatsEngine = None
TacticalMapRenderer = None
TrackingIO = None
compose_final_frame = None
render_video_overlay = None


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


from infra.configs import Config


Rotation = Literal["clockwise", "counterclockwise", "none"]

DEFAULT_FPS = 30
DEFAULT_OUTPUT_SIZE = (1600, 900)
LEFT_WIDTH_RATIO = 0.64
VIDEO_PREVIEW_HEIGHT_RATIO = 0.50
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


class SequentialVideoReader:
    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open source video {video_path}")
        self.next_frame_id: Optional[int] = None

    @property
    def frame_count(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read(self, frame_id: int) -> np.ndarray:
        if self.next_frame_id != frame_id:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))

        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_id} from {self.video_path}")

        self.next_frame_id = frame_id + 1
        return frame

    def release(self) -> None:
        self.capture.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render tracking JSONL over the source video.")
    parser.add_argument("--video", type=Path, default=Config.VIDEO_DIR / "video1.mp4", help="Source video path.")
    parser.add_argument("--tracking", type=Path, default=Config.TRACKING_DIR / "tracking.jsonl", help="Tracking JSONL path.")
    parser.add_argument("--output-dir", type=Path, default=Config.OUTPUT_DIR / "tracking_visualization", help="Output directory.")
    parser.add_argument("--output-video", type=Path, default=None, help="Output MP4 path. Defaults inside --output-dir.")
    parser.add_argument("--rotation", choices=("clockwise", "counterclockwise", "none"), default="clockwise", help="Video rotation applied before rendering.")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Output video FPS.")
    parser.add_argument("--start-frame", type=int, default=0, help="First source frame to render.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of tracking frames to render.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _load_cv_modules()
    _load_project_modules()
    output_video = args.output_video or args.output_dir / "tracking_visualization.mp4"
    render_tracking_visualization(
        video_path=args.video,
        tracking_path=args.tracking,
        output_dir=args.output_dir,
        output_video=output_video,
        rotation=args.rotation,
        fps=float(args.fps),
        start_frame=max(0, int(args.start_frame)),
        max_frames=args.max_frames,
    )


def _load_cv_modules() -> None:
    global cv2, np
    if cv2 is not None and np is not None:
        return
    import cv2 as cv2_module
    import numpy as np_module

    cv2 = cv2_module
    np = np_module


def _load_project_modules() -> None:
    global DashboardRenderer, EventDetector, FieldStyle, FrameResult, Point2D
    global StatsEngine, TacticalMapRenderer, TrackingIO, compose_final_frame, render_video_overlay
    if TrackingIO is not None:
        return

    from analysis.event_detector import EventDetector as EventDetectorClass
    from analysis.stats_engine import StatsEngine as StatsEngineClass
    from domain.entities import FrameResult as FrameResultClass
    from domain.entities import Point2D as Point2DClass
    from io_utils.tracking_io import TrackingIO as TrackingIOClass
    from visualization.dashboard import DashboardRenderer as DashboardRendererClass
    from visualization.layout import compose_final_frame as compose_final_frame_func
    from visualization.tactical_map import FieldStyle as FieldStyleClass
    from visualization.tactical_map import TacticalMapRenderer as TacticalMapRendererClass
    from visualization.video_render import render_video_overlay as render_video_overlay_func

    DashboardRenderer = DashboardRendererClass
    EventDetector = EventDetectorClass
    FieldStyle = FieldStyleClass
    FrameResult = FrameResultClass
    Point2D = Point2DClass
    StatsEngine = StatsEngineClass
    TacticalMapRenderer = TacticalMapRendererClass
    TrackingIO = TrackingIOClass
    compose_final_frame = compose_final_frame_func
    render_video_overlay = render_video_overlay_func


def render_tracking_visualization(
    *,
    video_path: Path,
    tracking_path: Path,
    output_dir: Path,
    output_video: Path,
    rotation: Rotation,
    fps: float,
    start_frame: int,
    max_frames: Optional[int],
) -> None:
    _load_cv_modules()
    _load_project_modules()
    frames = TrackingIO(tracking_path).read_frame_results()
    frames = [frame for frame in frames if frame.frame_id >= start_frame]
    if max_frames is not None:
        frames = frames[: max(0, int(max_frames))]
    if not frames:
        raise RuntimeError(f"No frames found in {tracking_path} for start_frame={start_frame}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    reader = SequentialVideoReader(video_path)
    _warn_if_tracking_mismatch(tracking_path, video_path, reader.frame_count)

    output_w, output_h = DEFAULT_OUTPUT_SIZE
    left_w = int(round(output_w * LEFT_WIDTH_RATIO))
    dashboard_w = output_w - left_w
    video_h = int(round(output_h * VIDEO_PREVIEW_HEIGHT_RATIO))
    tactical_h = output_h - video_h
    video_size = (left_w, video_h)

    event_detector = EventDetector(default_fps=fps)
    stats_engine = StatsEngine()
    dashboard_renderer = DashboardRenderer()
    tactical_renderer = TacticalMapRenderer(
        style=FieldStyle(output_size=(left_w, tactical_h), mirror_x=True),
        trail_length=60,
    )

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        DEFAULT_OUTPUT_SIZE,
    )
    if not writer.isOpened():
        reader.release()
        raise RuntimeError(f"Could not open video writer for {output_video}")

    frame_first_path = output_dir / "frame_first.png"
    frame_event_path = output_dir / "frame_first_event.png"
    frame_last_path = output_dir / "frame_last.png"
    first_event_frame: Optional[np.ndarray] = None
    first_frame: Optional[np.ndarray] = None
    last_frame: Optional[np.ndarray] = None
    event_counts: dict[str, int] = {}

    try:
        for source_frame in frames:
            raw_video, transform = _read_render_frame(reader, source_frame.frame_id, video_size, rotation=rotation)
            frame = _transform_frame_pixels(source_frame, transform, rotation)
            frame_events = event_detector.detect(frame)
            stats = stats_engine.update(frame, frame_events)
            match_time_seconds = frame.timestamp_s if frame.timestamp_s is not None else frame.frame_id / fps

            for event in frame_events.eventos:
                event_counts[event.type] = event_counts.get(event.type, 0) + 1

            video_overlay = render_video_overlay(raw_video, frame, frame_events)
            tactical_map = tactical_renderer.render(frame, frame_events)
            dashboard = dashboard_renderer.render(stats, frame_events, match_time_seconds, dashboard_w, output_h)
            final_frame = compose_final_frame(
                video_overlay,
                tactical_map,
                dashboard,
                output_size=DEFAULT_OUTPUT_SIZE,
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
        reader.release()
        writer.release()

    if first_frame is not None:
        cv2.imwrite(str(frame_first_path), first_frame)
    if first_event_frame is not None:
        cv2.imwrite(str(frame_event_path), first_event_frame)
    if last_frame is not None:
        cv2.imwrite(str(frame_last_path), last_frame)

    print(f"OK first frame: {frame_first_path}")
    print(f"OK first event frame: {frame_event_path if first_event_frame is not None else 'none'}")
    print(f"OK last frame: {frame_last_path}")
    print(f"OK video: {output_video}")
    print(f"frames: {len(frames)}")
    print(f"event_counts: {event_counts}")


def _warn_if_tracking_mismatch(tracking_path: Path, video_path: Path, video_frame_count: int) -> None:
    metadata = TrackingIO(tracking_path).read_metadata()
    if not metadata:
        print(f"WARNING: no tracking metadata found for {tracking_path}")
        return

    meta_video_name = metadata.get("video_name")
    if meta_video_name and meta_video_name != video_path.name:
        print(f"WARNING: tracking was generated for {meta_video_name}, rendering {video_path.name}")

    meta_frame_count = metadata.get("frame_count")
    if meta_frame_count and video_frame_count and int(meta_frame_count) != int(video_frame_count):
        print(f"WARNING: tracking frame_count={meta_frame_count}, video frame_count={video_frame_count}")


def _read_render_frame(
    reader: SequentialVideoReader,
    frame_id: int,
    output_size: tuple[int, int],
    *,
    rotation: Rotation,
) -> tuple[np.ndarray, VideoFrameTransform]:
    frame = reader.read(frame_id)
    source_h, source_w = frame.shape[:2]
    rotated = _rotate_frame(frame, rotation)
    rendered, transform = _letterbox_frame(rotated, output_size, source_width=source_w, source_height=source_h)
    return rendered, transform


def _rotate_frame(frame: np.ndarray, rotation: Rotation) -> np.ndarray:
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
    return rendered, VideoFrameTransform(
        source_width=source_width,
        source_height=source_height,
        rotated_width=frame_w,
        rotated_height=frame_h,
        scale=scale,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def _transform_frame_pixels(frame: FrameResult, transform: VideoFrameTransform, rotation: Rotation) -> FrameResult:
    robots = [
        replace(robot, position_pixel=_transform_point(robot.position_pixel, transform, rotation))
        for robot in frame.robots
    ]
    ball = None
    if frame.ball is not None:
        ball = replace(frame.ball, position_pixel=_transform_point(frame.ball.position_pixel, transform, rotation))
    return replace(frame, robots=robots, ball=ball, repositions=list(frame.repositions))


def _transform_point(point: Point2D, transform: VideoFrameTransform, rotation: Rotation) -> Point2D:
    x, y = _rotate_point(float(point.x), float(point.y), transform, rotation)
    x = transform.offset_x + x * transform.scale
    y = transform.offset_y + y * transform.scale
    return Point2D(x, y, is_metric=False)


def _rotate_point(x: float, y: float, transform: VideoFrameTransform, rotation: Rotation) -> tuple[float, float]:
    if rotation == "clockwise":
        return transform.source_height - 1.0 - y, x
    if rotation == "counterclockwise":
        return y, transform.source_width - 1.0 - x
    if rotation == "none":
        return x, y
    raise ValueError(f"Unsupported video rotation: {rotation}")


if __name__ == "__main__":
    main()
