from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import domain.entities as domain_entities
from analysis.event_detector import EventDetector
from analysis.stats_engine import StatsEngine
from domain.entities import FrameResult
from infra.configs import Config
from infra.event_bus import EventBus
from io_utils.tracking_io import TrackingIO
from visualization.dashboard import DashboardRenderer
from visualization.layout import compose_final_frame
from visualization.tactical_map import TacticalMapRenderer
from visualization.video_render import VideoOverlayRenderer


if not hasattr(domain_entities, "Team"):
    from dataclasses import dataclass, field

    @dataclass
    class Team:
        name: str
        color: str
        score: int = 0
        robots: List[Robot] = field(default_factory=list)

    domain_entities.Team = Team


VIDEO_W = 1280
VIDEO_H = 1800


def main() -> None:
    output_dir = Config.OUTPUT_DIR / "analysis_visual_tracking"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Config.OUTPUT_DIR / "analysis_visual_tracking.mp4"

    bus = EventBus()
    latest_outputs: Dict[str, Optional[np.ndarray]] = {
        "tactical_map": None,
        "video_overlay": None,
        "dashboard": None,
    }

    bus.subscribe("tactical_map", lambda image: _capture(latest_outputs, "tactical_map", image))
    bus.subscribe("video_overlay", lambda image: _capture(latest_outputs, "video_overlay", image))
    bus.subscribe("dashboard", lambda image: _capture(latest_outputs, "dashboard", image))

    EventDetector(bus)
    StatsEngine(bus)
    TacticalMapRenderer(bus)
    VideoOverlayRenderer(bus, include_tactical_map=False)
    DashboardRenderer(bus)

    frames = _build_frame_results()
    video_writer: Optional[cv2.VideoWriter] = None
    final_frame: Optional[np.ndarray] = None

    try:
        for frame in frames:
            bus.publish("frame_result_raw", frame)
            bus.publish("frame_result", frame)
            bus.publish("video_frame", _dummy_video_frame(frame.frame_id))

            tactical_map = _require_output(latest_outputs, "tactical_map")
            video_overlay = _require_output(latest_outputs, "video_overlay")
            dashboard = _require_output(latest_outputs, "dashboard")
            final_frame = compose_final_frame(video_overlay, tactical_map, dashboard)

            if video_writer is None:
                height, width = final_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (width, height))
                if not video_writer.isOpened():
                    raise RuntimeError(f"Could not open video writer for {video_path}")

            output_path = output_dir / f"analysis_frame_{frame.frame_id:03d}.png"
            cv2.imwrite(str(output_path), final_frame)
            video_writer.write(final_frame)
    finally:
        if video_writer is not None:
            video_writer.release()

    if final_frame is None:
        raise RuntimeError("No frames were rendered.")

    print(f"analysis visual smoke frames written to: {output_dir}")
    print(f"analysis visual smoke video written to: {video_path}")
    print(f"frames: {len(frames)}")
    print(f"final_frame: {final_frame.shape}")


def _build_frame_results() -> List[FrameResult]:
    tracking_path = Config.TRACKING_DIR / "tracking_with_metric.jsonl"
    frames = TrackingIO(tracking_path).read_frame_results()
    if not frames:
        tracking_path = Config.TRACKING_DIR / "tracking.jsonl"
        frames = TrackingIO(tracking_path).read_frame_results()
    if not frames:
        raise RuntimeError("No rows found in tracking_with_metric.jsonl or tracking.jsonl.")
    return frames


def _dummy_video_frame(frame_id: int) -> np.ndarray:
    frame = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
    frame[:, :, 0] = np.linspace(24, 90, VIDEO_W, dtype=np.uint8)
    frame[:, :, 1] = np.linspace(36, 110, VIDEO_H, dtype=np.uint8)[:, None]
    frame[:, :, 2] = np.uint8((42 + frame_id * 3) % 256)
    cv2.rectangle(frame, (80, 70), (VIDEO_W - 80, VIDEO_H - 70), (80, 130, 80), 2)
    cv2.line(frame, (VIDEO_W // 2, 70), (VIDEO_W // 2, VIDEO_H - 70), (80, 130, 80), 1)
    cv2.putText(frame, "Analysis visual tracking", (90, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
    return frame


def _capture(outputs: Dict[str, Optional[np.ndarray]], key: str, image: np.ndarray) -> None:
    outputs[key] = image


def _require_output(outputs: Dict[str, Optional[np.ndarray]], key: str) -> np.ndarray:
    image = outputs[key]
    if image is None:
        raise RuntimeError(f"Renderer did not publish '{key}'.")
    return image


if __name__ == "__main__":
    main()
