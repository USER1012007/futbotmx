from math import hypot, atan2, degrees
from typing import Dict, List, Tuple, Optional
from collections import deque
from domain.entities import FrameResult, Robot, Ball, Point2D

class DataEnricher:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        # Tracker ID -> deque([(x, y, frame_id), ...])
        self.history: Dict[str, deque] = {}
        self.fps = 30.0

    def enrich(self, result: FrameResult):
        for robot in result.robots:
            if robot.tracker_id != -1 and robot.position_metric:
                self._enrich_robot(robot, result.frame_id)
        
        if result.ball and result.ball.position_metric:
            self._enrich_ball(result.ball, result.frame_id)

    def _get_smoothed_velocity(self, key: str, pos: Tuple[float, float], frame_id: int) -> Tuple[float, float, float]:
        if key not in self.history:
            self.history[key] = deque(maxlen=self.window_size)
        
        self.history[key].append((pos[0], pos[1], frame_id))
        
        if len(self.history[key]) < 2:
            return 0.0, 0.0, 0.0

        # Calcular velocidad basada en el promedio de la ventana
        start_x, start_y, start_frame = self.history[key][0]
        end_x, end_y, end_frame = self.history[key][-1]
        
        dt = max(1, end_frame - start_frame) / self.fps
        vx = (end_x - start_x) / dt
        vy = (end_y - start_y) / dt
        speed = hypot(vx, vy)
        
        return vx, vy, speed

    def _enrich_robot(self, robot: Robot, frame_id: int):
        key = f"robot_{robot.tracker_id}"
        pos = (robot.position_metric.x, robot.position_metric.y)
        
        vx, vy, speed = self._get_smoothed_velocity(key, pos, frame_id)
        
        robot.speed = speed
        # Dirección es la inversa del vector de trayectoria
        robot.angle = degrees(atan2(-vy, -vx))
            
    def _enrich_ball(self, ball: Ball, frame_id: int):
        key = "ball"
        pos = (ball.position_metric.x, ball.position_metric.y)
        
        vx, vy, speed = self._get_smoothed_velocity(key, pos, frame_id)
        
        ball.speed_cm_s = speed
        ball.direction_vector = (vx, vy)
