import cv2
import supervision as sv
import torch
import numpy as np
from ultralytics.models.sam import SAM3SemanticPredictor
from domain.entities import FrameResult, Robot, Ball
from infra.configs import Config
from typing import Optional

class SegmentationEngine:
    def __init__(self, cfg: Config):
        # Force CPU device for SAM3 to prevent OOM
        overrides = dict(conf=0.25, task="segment", mode="predict", model=cfg.SAM_MODEL_NAME, device='cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        self.cfg = cfg

    def process_frame(self, frame: cv2.typing.MatLike, frame_id: int) -> FrameResult:
        self.predictor.set_image(frame)
        
        results = self.predictor(text=["soccer robot", "ball"])[0]
        detections = sv.Detections.from_ultralytics(results)
        
        robots = []
        best_ball = None
        best_ball_conf = 0.0
        
        for i in range(len(detections)):
            class_id = int(detections.class_id[i]) # 0: robot, 1: ball
            xyxy = detections.xyxy[i]
            conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2
            
            if class_id == 0: 
                robots.append(Robot(id=f"robot_{i}", team_id="unknown", position=(float(center_x), float(center_y))))
            elif class_id == 1:
                if conf > best_ball_conf:
                    best_ball = Ball(id="ball", position=(float(center_x), float(center_y)))
                    best_ball_conf = conf
                
        return FrameResult(frame_id=frame_id, robots=robots, ball=best_ball)

