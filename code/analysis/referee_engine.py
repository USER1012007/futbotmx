from domain.events import CollisionEvent, RemoveRobotEvent
from infra.event_bus import EventBus

class RefereeEngine:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        # Suscribirse a eventos de bajo nivel
        self.event_bus.subscribe("frame_events", self._handle_events)

    def _handle_events(self, events):
        for event in events.eventos:
            if isinstance(event, CollisionEvent):
                # Ejemplo de regla: penalizar si hay colisión
                # En un futuro, podrías agregar lógica para saber qué robot fue el culpable
                pass
            
            if isinstance(event, RemoveRobotEvent):
                # Lógica para marcar robot como penalizado
                robot_id = event.robot.id
                print(f"[Referee] Marcando robot {robot_id} como penalizado.")
                # Aquí actualizaríamos el estado del robot en un GameStateManager
