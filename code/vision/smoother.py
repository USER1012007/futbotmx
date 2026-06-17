from domain.entities import Point2D

class PositionSmoother:
    def __init__(self, alpha: float = 0.3):
        """
        alpha: Factor de suavizado (0.0 a 1.0). 
        Valores menores = más suave pero con más inercia.
        Valores mayores = más responsivo pero con más parpadeo.
        """
        self.alpha = alpha
        self.history = {}

    def smooth(self, entity_id: str, current_pos: Point2D) -> Point2D:
        if current_pos is None:
            return None
            
        # Si es la primera vez que vemos esta entidad, guardamos su posición actual
        if entity_id not in self.history:
            self.history[entity_id] = current_pos
            return current_pos
        
        prev_pos = self.history[entity_id]
        
        # Aplicar suavizado exponencial (EMA)
        smooth_x = self.alpha * current_pos.x + (1 - self.alpha) * prev_pos.x
        smooth_y = self.alpha * current_pos.y + (1 - self.alpha) * prev_pos.y
        
        smoothed_point = Point2D(x=smooth_x, y=smooth_y, is_metric=current_pos.is_metric)
        
        # Actualizar el historial
        self.history[entity_id] = smoothed_point
        return smoothed_point
