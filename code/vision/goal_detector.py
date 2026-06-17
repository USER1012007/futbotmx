import cv2
import numpy as np

class GoalDetector:
    def __init__(self):
        # Rangos HSV para amarillo y azul
        self.yellow_lower = np.array([20, 100, 100])
        self.yellow_upper = np.array([30, 255, 255])
        self.blue_lower = np.array([100, 150, 50])
        self.blue_upper = np.array([140, 255, 255])

    def get_goal_positions(self, frame: np.ndarray):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        
        yellow_pos = self._get_centroid(yellow_mask)
        blue_pos = self._get_centroid(blue_mask)
        
        return yellow_pos, blue_pos

    def _get_centroid(self, mask):
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None
