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
        
        # Persistencia para frames donde no se detecten las porterías
        self.last_corners: Optional[np.ndarray] = None

        # Destino siempre estático: TL, TR, BR, BL
        self.dst_points = np.array([
            [0, 0],
            [self.field_width_cm, 0],
            [self.field_width_cm, self.field_height_cm],
            [0, self.field_height_cm]
        ], dtype=np.float32)

    def extract_raw_corners(self, mask: np.ndarray) -> Optional[np.ndarray]:
        if mask is None or mask.size == 0:
            return None
        if mask.ndim == 3:
            mask = mask[0]
            
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        field_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(field_contour, True)
        corners = cv2.approxPolyDP(field_contour, 0.05 * peri, True)
        
        if len(corners) == 4:
            return corners.reshape(4, 2).astype(np.float32)
        return None

    def _sort_corners_by_goals(self, corners: np.ndarray, yg: tuple, bg: tuple) -> np.ndarray:
        yg = np.array(yg)
        bg = np.array(bg)

        # Vector longitudinal de la cancha (de Amarillo a Azul)
        v_yb = bg - yg

        # Proyección para separar Izquierda (Amarillo) de Derecha (Azul)
        projections = np.dot(corners - yg, v_yb)
        sorted_indices = np.argsort(projections)
        
        left_corners = corners[sorted_indices[:2]]
        right_corners = corners[sorted_indices[2:]]

        # Producto cruz bidimensional para separar Arriba de Abajo
        def cross_product(p):
            v_p = p - yg
            return v_yb[0] * v_p[1] - v_yb[1] * v_p[0]

        # Ordenar Izquierda: TL, BL
        if cross_product(left_corners[0]) < cross_product(left_corners[1]):
            tl, bl = left_corners[0], left_corners[1]
        else:
            tl, bl = left_corners[1], left_corners[0]

        # Ordenar Derecha: TR, BR
        if cross_product(right_corners[0]) < cross_product(right_corners[1]):
            tr, br = right_corners[0], right_corners[1]
        else:
            tr, br = right_corners[1], right_corners[0]

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _match_corners_to_previous(self, current: np.ndarray, previous: np.ndarray) -> np.ndarray:
        ordered = np.zeros_like(current)
        used = set()
        for i, p_prev in enumerate(previous):
            distances = [np.linalg.norm(c - p_prev) if j not in used else float('inf') for j, c in enumerate(current)]
            best_idx = int(np.argmin(distances))
            ordered[i] = current[best_idx]
            used.add(best_idx)
        return ordered

    def update_homography_from_mask(self, mask: np.ndarray, yellow_goal: Optional[tuple] = None, blue_goal: Optional[tuple] = None):
        raw_corners = self.extract_raw_corners(mask)
        if raw_corners is None:
            return

        sorted_corners = None
        
        # Si vemos ambas porterías, calibramos la orientación real
        if yellow_goal and blue_goal:
            sorted_corners = self._sort_corners_by_goals(raw_corners, yellow_goal, blue_goal)
            self.last_corners = sorted_corners
            
        # Si no, mantenemos la orientación mapeando a los últimos vértices conocidos
        elif self.last_corners is not None:
            sorted_corners = self._match_corners_to_previous(raw_corners, self.last_corners)
            self.last_corners = sorted_corners

        if sorted_corners is not None:
            # cv2.RANSAC no es estrictamente necesario para 4 puntos, pero da estabilidad.
            new_H, _ = cv2.findHomography(sorted_corners, self.dst_points, 0)
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
