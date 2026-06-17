from infra.configs import Config
from vision.pipeline import Pipeline
from analysis.pipeline import AnalysisPipeline
import sys
from pathlib import Path

def main():
    cfg = Config()
    video_path = cfg.BASE_DIR / "data/videos/video1.mov"
    
    if not video_path.exists():
        print(f"Video/Image file not found at {video_path}")
        sys.exit(1)
        
    print(f"Starting system...")
    pipeline = Pipeline(cfg, str(video_path))
    analysis = AnalysisPipeline(pipeline.event_bus)
    
    pipeline.run()
    print("Pipeline finished.")

if __name__ == "__main__":
    main()
