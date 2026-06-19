import supervision as sv
from io_utils.video_source import VideoSource
from vision.segmentation import SegmentationEngine
from vision.homography import HomographyEngine
from vision.goal_detector import GoalDetector
from vision.smoother import PositionSmoother
from vision.team_assignment import BallMetricStabilizer, TeamAssigner
from pathlib import Path

from io_utils.tracking_io import TrackingIO
from infra.configs import Config
from infra.event_bus import EventBus


class Pipeline:
    def __init__(self, cfg: Config, video_path: str | Path, tracking_path: str | Path | None = None, max_frames: int | None = None):
        self.cfg = cfg
        self.video_path = Path(video_path)
        self.max_frames = max_frames
        self.video_source = VideoSource(video_path)
        self.segmentation = SegmentationEngine(cfg)
        self.homography = HomographyEngine(cfg)
        self.goal_detector = GoalDetector()
        self.smoother = PositionSmoother(alpha=0.25)
        self.tracking_io = TrackingIO(Path(tracking_path) if tracking_path is not None else cfg.TRACKING_DIR / "tracking.jsonl")
        self.event_bus = EventBus()
        self._fps = self._detect_fps()
        self.team_assigner = TeamAssigner()
        self.ball_stabilizer = BallMetricStabilizer(self._fps)

        # Annotators
        self.mask_annotator = sv.MaskAnnotator()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def run(self):
        self.tracking_io.reset()
        self.tracking_io.save_metadata(
            {
                "video_name": self.video_path.name,
                "video_path": str(self.video_path),
                "frame_count": self.video_source.frame_count,
                "fps": self._fps,
                "max_frames": self.max_frames,
                "sam_device": self.cfg.SAM_DEVICE,
                "sam_imgsz": self.cfg.SAM_IMGSZ,
            }
        )
        frame_id = 0
        while True:
            if self.max_frames is not None and frame_id >= self.max_frames:
                break

            frame = self.video_source.get_frame()
            if frame is None:
                break
            
            # 1. Segmentación
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

            result.timestamp_s = frame_id / self._fps if self._fps else None
            self.team_assigner.assign(result, frame)

            # Suavizado con IDs estables. Después de suavizar píxeles, reproyectamos
            # para que las métricas correspondan a la posición final escrita.
            for robot in result.robots:
                robot.position_pixel = self.smoother.smooth(robot.id, robot.position_pixel)
                if self.homography.H is not None:
                    robot.position_metric = self.homography.project_point(robot.position_pixel)
            if result.ball:
                result.ball.position_pixel = self.smoother.smooth(result.ball.id, result.ball.position_pixel)
                if self.homography.H is not None:
                    result.ball.position_metric = self.homography.project_point(result.ball.position_pixel)
            self.ball_stabilizer.stabilize(result)
            
            # 4. Publicar evento y persistencia
            self.event_bus.publish("frame_processed", result)
            self.tracking_io.save_frame_data(frame_id, result.to_dict())

            frame_id += 1
            if frame_id % 30 == 0:
                print(f"Processed {frame_id} frames")

        self.video_source.release()

    def _detect_fps(self) -> float:
        fps = self.video_source.fps
        if fps and fps > 0:
            return float(fps)
        return float(getattr(self.cfg, "FPS_LIMIT", 30) or 30)
