from infra.configs import Config
from vision.pipeline import Pipeline
import sys
from pathlib import Path

def main():
    cfg = Config()
    video_path = cfg.VIDEO_DIR / "video1.mp4"
    
    if not video_path.exists():
        print(f"Video file not found at {video_path}")
        sys.exit(1)
        
    print(f"Starting pipeline with {video_path}...")
    pipeline = Pipeline(cfg, str(video_path))
    pipeline.run()
    print("Pipeline finished.")

if __name__ == "__main__":
    main()
