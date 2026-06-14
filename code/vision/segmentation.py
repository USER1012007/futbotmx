import cv2
import supervision as sv
import torch
import numpy as np
from trackers import ByteTrackTracker
from ultralytics.models.sam import SAM3SemanticPredictor
from domain.entities import FrameResult, Robot, Ball, Point2D
from infra.configs import Config
from typing import Optional

class SegmentationEngine:
    def __init__(self, cfg: Config):
        # Force CPU device for SAM3 to prevent OOM
        model_path = str(cfg.BASE_DIR / "sam3.pt")
        overrides = dict(
            conf=0.3, 
            task="segment", 
            mode="predict", 
            model=model_path, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            imgsz=1280
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.cfg = cfg
        # Inicialización del nuevo tracker
        self.tracker = ByteTrackTracker(
            lost_track_buffer=60, 
            track_activation_threshold=0.25, 
            minimum_iou_threshold=0.3
        )
        self.last_positions = {}

    def _get_distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def process_frame(self, frame: cv2.typing.MatLike, frame_id: int) -> FrameResult:
        self.predictor.set_image(frame)
        
        results = self.predictor(text=["soccer robot", "ball", "green soccer field"])[0]
        detections = sv.Detections.from_ultralytics(results)
        
        detections = detections.with_nms(threshold=0.3)
        detections = detections[detections.confidence > 0.4]
        detections = self.tracker.update(detections)
        
        robot_mask = detections.class_id == 0
        ball_mask = detections.class_id == 1
        field_mask = detections.class_id == 2
        
        robots_det = detections[robot_mask]
        ball_det = detections[ball_mask]
        
        # Procesamiento morfológico de la máscara
        field_mask_arr = None
        if len(detections[field_mask]) > 0:
            field_mask_raw = detections.mask[field_mask].astype(np.uint8)
            # Combinar máscaras si hay múltiples fragmentos
            mask_combined = np.any(field_mask_raw, axis=0).astype(np.uint8)
            kernel = np.ones((15, 15), np.uint8)
            field_mask_arr = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
            
        robots = []
        for i in range(len(robots_det)):
            tracker_id = int(robots_det.tracker_id[i])
            xyxy = robots_det.xyxy[i]
            pos = Point2D(x=(xyxy[0] + xyxy[2]) / 2, y=(xyxy[1] + xyxy[3]) / 2, is_metric=False)
            robots.append(Robot(id=f"robot_{tracker_id}", tracker_id=tracker_id, team_id="unknown", position_pixel=pos))
            self.last_positions[tracker_id] = (pos.x, pos.y)
            
        best_ball = None
        if len(ball_det) > 0:
            candidate_idx = 0
            if "ball" in self.last_positions:
                last_pos = self.last_positions["ball"]
                min_dist = float('inf')
                for i in range(len(ball_det)):
                    xyxy = ball_det.xyxy[i]
                    center = ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
                    dist = self._get_distance(center, last_pos)
                    if dist < min_dist:
                        min_dist = dist
                        candidate_idx = i
            
            tracker_id = int(ball_det.tracker_id[candidate_idx])
            xyxy = ball_det.xyxy[candidate_idx]
            pos = Point2D(x=(xyxy[0] + xyxy[2]) / 2, y=(xyxy[1] + xyxy[3]) / 2, is_metric=False)
            best_ball = Ball(id=f"ball_{tracker_id}", tracker_id=tracker_id, position_pixel=pos)
            self.last_positions["ball"] = (pos.x, pos.y)
                
        return FrameResult(frame_id=frame_id, robots=robots, ball=best_ball, field_mask=field_mask_arr)

