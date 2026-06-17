from typing import Dict, List, Optional, Tuple, Iterable
import math

class EventDeduper:
    def __init__(self, cooldowns: Dict[str, int]):
        self.cooldowns = cooldowns
        self.last_emitted_events: Dict[Tuple, int] = {}

    def clear(self):
        self.last_emitted_events.clear()

    def dedupe(self, events: List[object], frame: int) -> List[object]:
        deduped: List[object] = []
        for event in events:
            event_type = getattr(event, "type", event.__class__.__name__)
            cooldown = self.cooldowns.get(event_type, 0)
            if cooldown <= 0:
                deduped.append(event)
                continue

            key = self._dedupe_key(event)
            last_frame = self.last_emitted_events.get(key)
            if last_frame is not None and frame - last_frame < cooldown:
                continue

            self.last_emitted_events[key] = frame
            deduped.append(event)
        return deduped

    def _dedupe_key(self, event: object) -> Tuple:
        event_type = getattr(event, "type", event.__class__.__name__)

        if event_type == "pase":
            return (
                event_type,
                getattr(event, "from_robot_id", getattr(event, "from_id", None)),
                getattr(event, "to_robot_id", getattr(event, "to", None)),
            )

        if event_type == "colision":
            robot_ids = sorted(getattr(robot, "id", str(robot)) for robot in getattr(event, "robots", []))
            return (event_type, tuple(robot_ids))

        if event_type in {"gol_valido", "gol_invalido"}:
            return (event_type, self._team_name(getattr(event, "team", "")))

        if event_type in {"robot_detenido", "sacar_robot", "reposicion_robot"}:
            robot = getattr(event, "robot", None)
            return (event_type, getattr(robot, "id", None), self._position_bucket(self._event_position(event)))

        if event_type == "fuera_de_cancha":
            return (
                event_type,
                getattr(event, "object_type", getattr(event, "object", None)),
                getattr(event, "object_id", None),
            )

        if event_type == "reposicion_balon":
            return (
                event_type,
                getattr(event, "reason", None),
                self._position_bucket(self._event_position(event)),
            )

        if event_type == "panic":
            return (event_type, getattr(event, "reason", None))

        return (event_type, self._position_bucket(self._event_position(event)))

    @staticmethod
    def _event_position(event: object) -> Optional[Tuple[float, float]]:
        for attr in ("position_cm", "position", "last_position_cm", "target_position_cm", "to_position_cm"):
            value = getattr(event, attr, None)
            if value is not None:
                return EventDeduper._tuple_or_none(value)
        return None

    @staticmethod
    def _position_bucket(position: Optional[Tuple[float, float]], bucket_cm: float = 10.0) -> Optional[Tuple[int, int]]:
        if position is None:
            return None
        return (int(round(position[0] / bucket_cm)), int(round(position[1] / bucket_cm)))

    @staticmethod
    def _team_name(team: object) -> str:
        if isinstance(team, str):
            return team.lower()
        return getattr(team, "name", str(team)).lower()
    
    @staticmethod
    def _tuple_or_none(value: object) -> Optional[Tuple[float, float]]:
        if value is None:
            return None
        if isinstance(value, dict):
            return (float(value["x"]), float(value["y"]))
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return (float(value[0]), float(value[1]))
        return None
