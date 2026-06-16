from math import hypot, atan2, degrees
from typing import Dict, Tuple, Optional
from domain.entities import FrameResult, Robot, Ball, Point2D

class DataEnricher:
    def __init__(self):
        # Tracker ID -> (last_x, last_y, last_frame_id)
        self.history: Dict[str, Tuple[float, float, int]] = {}
        self.fps = 30.0  # Asumimos 30 FPS para cálculo de velocidad

    def enrich(self, result: FrameResult):
        # Enriquecer Robots
        for robot in result.robots:
            if robot.tracker_id != -1 and robot.position_metric:
                self._enrich_robot(robot, result.frame_id)
        
        # Enriquecer Ball
        if result.ball and result.ball.position_metric:
            self._enrich_ball(result.ball, result.frame_id)

    def _enrich_robot(self, robot: Robot, frame_id: int):
        key = f"robot_{robot.tracker_id}"
        pos = (robot.position_metric.x, robot.position_metric.y)
        
        if key in self.history:
            prev_x, prev_y, prev_frame = self.history[key]
            dt = max(1, frame_id - prev_frame) / self.fps
            dist = hypot(pos[0] - prev_x, pos[1] - prev_y)
            
            robot.speed = dist / dt  # cm/s
            robot.angle = degrees(atan2(pos[1] - prev_y, pos[0] - prev_x))
            
        self.history[key] = (pos[0], pos[1], frame_id)

    def _enrich_ball(self, ball: Ball, frame_id: int):
        key = "ball"
        pos = (ball.position_metric.x, ball.position_metric.y)
        
        if key in self.history:
            prev_x, prev_y, prev_frame = self.history[key]
            dt = max(1, frame_id - prev_frame) / self.fps
            dist = hypot(pos[0] - prev_x, pos[1] - prev_y)
            
            ball.speed_cm_s = dist / dt
            ball.direction_vector = (pos[0] - prev_x, pos[1] - prev_y)
            
        self.history[key] = (pos[0], pos[1], frame_id)
