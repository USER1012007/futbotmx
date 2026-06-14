import cv2
from io_utils.video_source import VideoSource
from vision.segmentation import SegmentationEngine
from vision.homography import HomographyEngine
from io_utils.tracking_io import TrackingIO
from infra.configs import Config
from pathlib import Path

class Pipeline:
    def __init__(self, cfg: Config, video_path: str):
        self.cfg = cfg
        self.video_source = VideoSource(video_path)
        self.segmentation = SegmentationEngine(cfg)
        self.homography = HomographyEngine(cfg)
        self.tracking_io = TrackingIO(cfg.TRACKING_DIR / "tracking.jsonl")

    def run(self):
        frame_id = 0
        while True:
            frame = self.video_source.get_frame()
            if frame is None:
                break
            
            # 1. Segmentación
            result = self.segmentation.process_frame(frame, frame_id)
            
            # 2. Homografía (Si hay máscara, recalibramos)
            if result.field_mask is not None:
                self.homography.update_homography_from_mask(result.field_mask)
            
            # 3. Proyección métrica (si H está lista)
            if self.homography.H is not None:
                for robot in result.robots:
                    robot.position_metric = self.homography.project_point(robot.position_pixel)
                if result.ball:
                    result.ball.position_metric = self.homography.project_point(result.ball.position_pixel)
            
            # 4. Persistencia
            self.tracking_io.save_frame_data(frame_id, result.to_dict())
            
            frame_id += 1
            if frame_id % 30 == 0:
                print(f"Processed {frame_id} frames")

        self.video_source.release()

