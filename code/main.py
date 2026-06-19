import argparse
import sys
from dataclasses import replace
from pathlib import Path

from infra.configs import Config


def parse_args() -> argparse.Namespace:
    cfg = Config()
    parser = argparse.ArgumentParser(description="Run FutBotMX tracking and analysis.")
    parser.add_argument("--video", type=Path, default=cfg.VIDEO_DIR / "video1.mp4", help="Input video path.")
    parser.add_argument("--tracking", type=Path, default=cfg.TRACKING_DIR / "tracking.jsonl", help="Output tracking JSONL path.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process.")
    parser.add_argument("--device", default=cfg.SAM_DEVICE, help="SAM device, for example cuda or cpu.")
    parser.add_argument("--imgsz", type=int, default=cfg.SAM_IMGSZ, help="SAM inference image size.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = replace(Config(), SAM_DEVICE=args.device, SAM_IMGSZ=args.imgsz)
    video_path = args.video

    if not video_path.exists():
        print(f"Video/Image file not found at {video_path}")
        sys.exit(1)

    print(f"Starting system with video: {video_path}")
    print(f"Tracking output: {args.tracking}")
    from analysis.pipeline import AnalysisPipeline
    from vision.pipeline import Pipeline

    pipeline = Pipeline(cfg, video_path, tracking_path=args.tracking, max_frames=args.max_frames)
    analysis = AnalysisPipeline(pipeline.event_bus)
    
    pipeline.run()
    print("Pipeline finished.")

if __name__ == "__main__":
    main()
