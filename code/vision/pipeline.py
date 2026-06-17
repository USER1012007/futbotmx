import cv2
import supervision as sv
from copy import deepcopy
from typing import Dict, Tuple
from io_utils.video_source import VideoSource
from vision.segmentation import SegmentationEngine
from vision.homography import HomographyEngine
from vision.goal_detector import GoalDetector
from vision.smoother import PositionSmoother
from io_utils.tracking_io import TrackingIO
from infra.configs import Config
from infra.event_bus import EventBus
from domain.field import FIELD_GEOMETRY
from pathlib import Path

MAX_BALL_SPEED_CM_S = 600.0

class Pipeline:
    def __init__(self, cfg: Config, video_path: str):
        self.cfg = cfg
        self.video_source = VideoSource(video_path)
        self.segmentation = SegmentationEngine(cfg)
        self.homography = HomographyEngine(cfg)
        self.goal_detector = GoalDetector()
        self.smoother = PositionSmoother(alpha=0.25)
        self.tracking_io = TrackingIO(cfg.TRACKING_DIR / "tracking.jsonl")
        self.event_bus = EventBus()
        self._tracker_identity: Dict[int, Tuple[str, str]] = {}
        self._team_counts: Dict[str, int] = {"allies": 0, "rivals": 0}
        self._fps = self._detect_fps()
        self._last_valid_ball = None
        self._last_valid_ball_frame: int | None = None
        
        # Annotators
        self.mask_annotator = sv.MaskAnnotator()
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def run(self):
        self.tracking_io.reset()
        frame_id = 0
        while True:
            frame = self.video_source.get_frame()
            if frame is None:
                break
            
            # 1. Segmentación
            result = self.segmentation.process_frame(frame, frame_id)
            
            # Suavizado de posiciones en espacio pixel
            for robot in result.robots:
                robot.position_pixel = self.smoother.smooth(robot.id, robot.position_pixel)
            if result.ball:
                result.ball.position_pixel = self.smoother.smooth(result.ball.id, result.ball.position_pixel)
            
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
            self._assign_stable_entities(result)
            self._stabilize_ball_metric(result)
            
            # 4. Publicar evento y persistencia
            self.event_bus.publish("frame_processed", result)
            self.tracking_io.save_frame_data(frame_id, result.to_dict())
            
            # Guardar frame visualmente
            # output_path = self.cfg.BASE_DIR / f"data/outputs/frame_{frame_id:04d}.jpg"
            # cv2.imwrite(str(output_path), frame)
            
            frame_id += 1
            if frame_id % 30 == 0:
                print(f"Processed {frame_id} frames")

        self.video_source.release()

    def _detect_fps(self) -> float:
        fps = self.video_source.fps
        if fps and fps > 0:
            return float(fps)
        return float(getattr(self.cfg, "FPS_LIMIT", 30) or 30)

    def _assign_stable_entities(self, result) -> None:
        for robot in result.robots:
            tracker_id = int(robot.tracker_id)
            if tracker_id not in self._tracker_identity:
                team_id = self._initial_team(robot)
                self._team_counts[team_id] += 1
                prefix = "A" if team_id == "allies" else "R"
                robot_id = f"{prefix}{self._team_counts[team_id]}"
                self._tracker_identity[tracker_id] = (robot_id, team_id)

            robot.id, robot.team_id = self._tracker_identity[tracker_id]

        if result.ball is not None:
            result.ball.id = "ball"

    def _initial_team(self, robot) -> str:
        position = getattr(robot, "position_metric", None)
        if position is not None:
            return "allies" if position.x <= FIELD_GEOMETRY.center_x_cm else "rivals"

        return "allies" if self._team_counts["allies"] <= self._team_counts["rivals"] else "rivals"

    def _stabilize_ball_metric(self, result) -> None:
        ball = result.ball
        if ball is None or ball.position_metric is None:
            if self._last_valid_ball is not None and self._last_valid_ball_frame is not None:
                missing_frames = result.frame_id - self._last_valid_ball_frame
                if missing_frames <= 8:
                    result.ball = deepcopy(self._last_valid_ball)
            return

        if self._last_valid_ball is not None and self._last_valid_ball.position_metric is not None:
            dt_frames = max(1, result.frame_id - (self._last_valid_ball_frame or result.frame_id))
            dt = dt_frames / self._fps if self._fps else dt_frames / 30.0
            prev = self._last_valid_ball.position_metric
            curr = ball.position_metric
            distance = ((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2) ** 0.5
            speed_cm_s = distance / max(dt, 1e-6)
            if speed_cm_s > MAX_BALL_SPEED_CM_S:
                result.ball = deepcopy(self._last_valid_ball)
                return

        self._last_valid_ball = deepcopy(result.ball)
        self._last_valid_ball_frame = result.frame_id
