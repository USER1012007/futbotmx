import cv2
import numpy as np
from typing import Optional
from domain.entities import Point2D
from infra.configs import Config

class HomographyEngine:
    def __init__(self, cfg: Config, smoothing_factor: float = 0.2):
        self.cfg = cfg
        self.H: Optional[np.ndarray] = None
        self.smoothing_factor = smoothing_factor
        self.field_width_cm = 243.0
        self.field_height_cm = 182.0
        # Puntos del campo real (en cm) en orden: TL, TR, BR, BL
        self.dst_points = np.array([
            [0, 0],
            [self.field_width_cm, 0],
            [self.field_width_cm, self.field_height_cm],
            [0, self.field_height_cm]
        ], dtype=np.float32)

    def extract_field_corners(self, mask: np.ndarray) -> Optional[np.ndarray]:
        if mask is None or mask.size == 0:
            return None

        # La máscara de supervisión suele ser (1, H, W) o (H, W)
        if mask.ndim == 3:
            mask = mask[0]
            
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Tomar el contorno más grande (el campo)
        field_contour = max(contours, key=cv2.contourArea)
        
        # Aproximar a polígono de 4 lados
        peri = cv2.arcLength(field_contour, True)
        corners = cv2.approxPolyDP(field_contour, 0.05 * peri, True)
        
        if len(corners) == 4:
            # Ordenar esquinas: top-left, top-right, bottom-right, bottom-left
            corners = corners.reshape(4, 2).astype(np.float32)
            rect = np.zeros((4, 2), dtype=np.float32)
            s = corners.sum(axis=1)
            rect[0] = corners[np.argmin(s)] # TL
            rect[2] = corners[np.argmax(s)] # BR
            diff = np.diff(corners, axis=1)
            rect[1] = corners[np.argmin(diff)] # TR
            rect[3] = corners[np.argmax(diff)] # BL
            return rect
        return None

    def update_homography_from_mask(self, mask: np.ndarray):
        src_points = self.extract_field_corners(mask)
        if src_points is not None:
            new_H, _ = cv2.findHomography(src_points, self.dst_points, cv2.RANSAC, 5.0)
            if new_H is not None:
                if self.H is None:
                    self.H = new_H
                else:
                    self.H = self.smoothing_factor * new_H + (1 - self.smoothing_factor) * self.H

    def project_point(self, point_pixel: Point2D) -> Optional[Point2D]:
        if self.H is None:
            return None
            
        pixel_array = np.array([[[point_pixel.x, point_pixel.y]]], dtype=np.float32)
        metric_array = cv2.perspectiveTransform(pixel_array, self.H)
        mx, my = metric_array[0][0]
        
        # Clamping
        mx = max(0.0, min(self.field_width_cm, mx))
        my = max(0.0, min(self.field_height_cm, my))
        
        return Point2D(x=float(mx), y=float(my), is_metric=True)
