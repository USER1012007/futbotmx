import cv2
import numpy as np
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
TEAM_CROP_HALF_SIZE_PX = 70
TEAM_DESCRIPTOR_ALPHA = 0.25
TEAM_REID_DISTANCE_THRESHOLD = 0.28
MAX_ROBOTS_PER_TEAM = 2
TEAM_SLOTS = {
    "allies": ("A1", "A2"),
    "rivals": ("R1", "R2"),
}

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
        self._identity_descriptors: Dict[str, np.ndarray] = {}
        self._tracker_descriptors: Dict[int, np.ndarray] = {}
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
            self._assign_stable_entities(result, frame)
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

    def _assign_stable_entities(self, result, frame) -> None:
        used_robot_ids = set()
        assigned_robots = []
        for robot in result.robots:
            tracker_id = int(robot.tracker_id)
            descriptor = self._robot_descriptor(frame, robot.position_pixel)

            identity = self._tracker_identity.get(tracker_id)
            if identity is None or identity[0] in used_robot_ids:
                identity = self._new_or_reidentified_identity(
                    robot,
                    descriptor,
                    used_robot_ids,
                )
                if identity is None:
                    continue
                self._tracker_identity[tracker_id] = identity

            robot.id, robot.team_id = identity
            used_robot_ids.add(robot.id)
            assigned_robots.append(robot)
            self._remember_robot_descriptor(tracker_id, robot.id, descriptor)

        result.robots = assigned_robots

        if result.ball is not None:
            result.ball.id = "ball"

    def _new_or_reidentified_identity(
        self,
        robot,
        descriptor: np.ndarray | None,
        used_robot_ids: set[str],
    ) -> Tuple[str, str] | None:
        match = self._match_known_identity(descriptor, used_robot_ids)
        if match is not None:
            return match

        return self._next_available_identity(self._initial_team(robot), used_robot_ids)

    def _match_known_identity(
        self,
        descriptor: np.ndarray | None,
        used_robot_ids: set[str],
    ) -> Tuple[str, str] | None:
        if descriptor is None or not self._identity_descriptors:
            return None

        best_robot_id = None
        best_distance = float("inf")
        for robot_id, known_descriptor in self._identity_descriptors.items():
            if robot_id in used_robot_ids:
                continue
            distance = self._descriptor_distance(descriptor, known_descriptor)
            if distance < best_distance:
                best_robot_id = robot_id
                best_distance = distance

        if best_robot_id is None or best_distance > TEAM_REID_DISTANCE_THRESHOLD:
            return None

        return best_robot_id, self._team_from_robot_id(best_robot_id)

    def _initial_team(self, robot) -> str:
        position = getattr(robot, "position_metric", None)
        if position is not None:
            return "allies" if position.x <= FIELD_GEOMETRY.center_x_cm else "rivals"

        return "allies"

    def _next_available_identity(
        self,
        preferred_team: str,
        used_robot_ids: set[str],
    ) -> Tuple[str, str] | None:
        for team_id in self._team_order(preferred_team, used_robot_ids):
            for robot_id in TEAM_SLOTS[team_id]:
                if robot_id not in used_robot_ids:
                    return robot_id, team_id
        return None

    def _team_order(self, preferred_team: str, used_robot_ids: set[str]) -> tuple[str, str]:
        other_team = "rivals" if preferred_team == "allies" else "allies"
        preferred_available = self._available_slots(preferred_team, used_robot_ids)
        other_available = self._available_slots(other_team, used_robot_ids)
        preferred_used = MAX_ROBOTS_PER_TEAM - preferred_available
        other_used = MAX_ROBOTS_PER_TEAM - other_available

        if preferred_available <= 0:
            return other_team, preferred_team
        if other_available > 0 and preferred_used > other_used:
            return other_team, preferred_team
        return preferred_team, other_team

    @staticmethod
    def _available_slots(team_id: str, used_robot_ids: set[str]) -> int:
        return sum(1 for robot_id in TEAM_SLOTS[team_id] if robot_id not in used_robot_ids)

    def _remember_robot_descriptor(
        self,
        tracker_id: int,
        robot_id: str,
        descriptor: np.ndarray | None,
    ) -> None:
        if descriptor is None:
            return

        previous_tracker = self._tracker_descriptors.get(tracker_id)
        if previous_tracker is not None:
            descriptor = self._blend_descriptor(previous_tracker, descriptor)
        self._tracker_descriptors[tracker_id] = descriptor

        previous_identity = self._identity_descriptors.get(robot_id)
        if previous_identity is not None:
            descriptor = self._blend_descriptor(previous_identity, descriptor)
        self._identity_descriptors[robot_id] = descriptor

    @staticmethod
    def _blend_descriptor(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        blended = (1.0 - TEAM_DESCRIPTOR_ALPHA) * previous + TEAM_DESCRIPTOR_ALPHA * current
        norm = np.linalg.norm(blended)
        return blended / norm if norm > 0 else blended

    @staticmethod
    def _team_from_robot_id(robot_id: str) -> str:
        return "allies" if robot_id.startswith("A") else "rivals"

    @staticmethod
    def _descriptor_distance(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-9:
            return 1.0
        similarity = float(np.dot(a, b) / denom)
        return 1.0 - max(-1.0, min(1.0, similarity))

    def _robot_descriptor(self, frame, point) -> np.ndarray | None:
        crop = self._crop_robot(frame, point)
        if crop is None or crop.size == 0:
            return None

        crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        h, s, v = cv2.split(hsv)
        green = (h >= 35) & (h <= 95) & (s >= 45) & (v >= 45)
        foreground = (~green) & (v >= 25)
        if int(np.count_nonzero(foreground)) < 80:
            foreground = v >= 25

        hue_hist = cv2.calcHist([h], [0], foreground.astype(np.uint8), [18], [0, 180]).flatten()
        sat_hist = cv2.calcHist([s], [0], foreground.astype(np.uint8), [8], [0, 256]).flatten()
        val_hist = cv2.calcHist([v], [0], foreground.astype(np.uint8), [8], [0, 256]).flatten()
        edges = cv2.Canny(gray, 60, 140)

        area = max(1, int(np.count_nonzero(foreground)))
        extras = np.array([
            np.count_nonzero(foreground) / foreground.size,
            np.count_nonzero(edges & foreground) / area,
            float(gray[foreground].mean() if np.any(foreground) else gray.mean()) / 255.0,
            float(gray[foreground].std() if np.any(foreground) else gray.std()) / 128.0,
        ], dtype=float)

        descriptor = np.concatenate([hue_hist, sat_hist, val_hist, extras]).astype(float)
        norm = np.linalg.norm(descriptor)
        return descriptor / norm if norm > 0 else None

    @staticmethod
    def _crop_robot(frame, point):
        if point is None:
            return None
        h, w = frame.shape[:2]
        cx = int(round(point.x))
        cy = int(round(point.y))
        x0 = max(0, cx - TEAM_CROP_HALF_SIZE_PX)
        y0 = max(0, cy - TEAM_CROP_HALF_SIZE_PX)
        x1 = min(w, cx + TEAM_CROP_HALF_SIZE_PX)
        y1 = min(h, cy + TEAM_CROP_HALF_SIZE_PX)
        if x1 <= x0 or y1 <= y0:
            return None
        return frame[y0:y1, x0:x1]

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
