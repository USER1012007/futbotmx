import cv2
from io_utils.video_source import VideoSource
from vision.segmentation import SegmentationEngine
from vision.homography import HomographyEngine
from vision.goal_detector import GoalDetector
from io_utils.tracking_io import TrackingIO
from infra.configs import Config
from infra.event_bus import EventBus
from pathlib import Path

class Pipeline:
    def __init__(self, cfg: Config, video_path: str):
        self.cfg = cfg
        self.video_source = VideoSource(video_path)
        self.segmentation = SegmentationEngine(cfg)
        self.homography = HomographyEngine(cfg)
        self.goal_detector = GoalDetector()
        self.tracking_io = TrackingIO(cfg.TRACKING_DIR / "tracking.jsonl")
        self.event_bus = EventBus()

    def run(self):
        frame_id = 0
        while True:
            frame = self.video_source.get_frame()
            if frame is None:
                break
            
            # 1. Segmentación con lógica de persistencia interna
            result = self.segmentation.process_frame(frame, frame_id)
            
            # 2. Detección de Porterías y Homografía
            yellow_goal, blue_goal = self.goal_detector.get_goal_positions(frame)
            if result.field_mask is not None:
                self.homography.update_homography_from_mask(result.field_mask, yellow_goal, blue_goal)
            
            # 3. Proyección métrica
            if self.homography.H is not None:
                for robot in result.robots:
                    robot.position_metric = self.homography.project_point(robot.position_pixel)
                if result.ball:
                    result.ball.position_metric = self.homography.project_point(result.ball.position_pixel)
            
            # 4. Publicar evento y persistencia
            self.event_bus.publish("frame_processed", result)
            self.tracking_io.save_frame_data(frame_id, result.to_dict())
            
            # Guardar frame visualmente (para depuración o registro)
            output_path = self.cfg.BASE_DIR / f"data/outputs/frame_{frame_id:04d}.jpg"
            cv2.imwrite(str(output_path), frame)
            
            frame_id += 1
            if frame_id % 30 == 0:
                print(f"Processed {frame_id} frames")

        self.video_source.release()

