import cv2
import supervision as sv
import torch
from ultralytics import YOLO, SAM
from typing import Optional
from domain.entities import FrameResult, Robot, Ball
from infra.configs import Config

class SegmentationEngine:
    def __init__(self, cfg: Config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_model = YOLO("yolov8x.pt") 
        self.sam_model = SAM(cfg.SAM_MODEL_NAME)
        self.cfg = cfg

    def process_frame(self, frame: cv2.typing.MatLike, frame_id: int) -> FrameResult:
        # 1. Detección con YOLO
        yolo_results = self.yolo_model(frame, device=self.device, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(yolo_results)
        
        # Filtrar por confianza
        mask = detections.confidence >= self.cfg.DETECTION_THRESHOLD
        detections = detections[mask]
        
        # 2. Conversión a estructuras del dominio
        robots = []
        ball = None
        
        for i in range(len(detections)):
            class_id = detections.class_id[i]
            xyxy = detections.xyxy[i]
            # Centroide del box
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2
            
            # Asumiendo mapeo de clases (ej. 0: robot, 1: ball)
            if class_id == 0:  # Robot
                robots.append(Robot(id=f"robot_{i}", team_id="unknown", position=(float(center_x), float(center_y))))
            elif class_id == 1: # Ball
                ball = Ball(id="ball", position=(float(center_x), float(center_y)))
                
        return FrameResult(frame_id=frame_id, robots=robots, ball=ball)
