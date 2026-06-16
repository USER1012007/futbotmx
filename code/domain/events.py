from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Union, Optional
from uuid import uuid4

from domain.entities import Robot, Team


EventSeverity = Literal["info", "warning", "danger", "success", "system"]

BallRepositionReason = Literal[
    "falta_de_progreso",
    "colision_area_penalti",
    "fuera_de_cancha",
]

RobotExpulsionReason = Literal[
    "toco_pared_cancha",
    "ingreso_area_penalti",
]

InvalidGoalReason = Literal[
    "robot_en_area_penalti",
]

PanicReason = Literal[
    "ruido_visual",
    "balon_no_detectado",
    "tracking_inestable",
]


@dataclass(kw_only=True)
class Event:
    frame: int
    timestamp_s: float
    position_cm: Optional[Tuple[float, float]] = None
    severity: EventSeverity = "info"
    confidence: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(kw_only=True)
class CollisionEvent(Event):
    robots: List[Robot]
    impact_speed_cm_s: Optional[float] = None
    type: Literal["colision"] = field(default="colision", init=False)


@dataclass(kw_only=True)
class PossessionEvent(Event):
    robot: Robot
    team: Team
    distance_cm: float
    type: Literal["posesion"] = field(default="posesion", init=False)


@dataclass(kw_only=True)
class PassEvent(Event):
    from_robot_id: str
    to_robot_id: str
    from_team: Optional[Team] = None
    to_team: Optional[Team] = None
    distance_cm: float = 0.0
    successful: bool = True
    type: Literal["pase"] = field(default="pase", init=False)


@dataclass(kw_only=True)
class OffCourtEvent(Event):
    object_type: Literal["balon", "robot"]
    object_id: Optional[str]
    last_position_cm: Tuple[float, float]
    type: Literal["fuera_de_cancha"] = field(default="fuera_de_cancha", init=False)


@dataclass(kw_only=True)
class RobotStoppedEvent(Event):
    robot: Robot
    frames_duration: int
    duration_s: Optional[float] = None
    type: Literal["robot_detenido"] = field(default="robot_detenido", init=False)


@dataclass(kw_only=True)
class BallRepositionEvent(Event):
    reason: BallRepositionReason
    from_position_cm: Optional[Tuple[float, float]]
    to_position_cm: Tuple[float, float]
    type: Literal["reposicion_balon"] = field(default="reposicion_balon", init=False)


@dataclass(kw_only=True)
class RobotRepositionEvent(Event):
    robot: Robot
    from_position_cm: Optional[Tuple[float, float]]
    target_position_cm: Tuple[float, float]
    type: Literal["reposicion_robot"] = field(default="reposicion_robot", init=False)


@dataclass(kw_only=True)
class GoalEvent(Event):
    team: Team
    velocity_cm_s: float
    scorer_robot: Optional[Robot] = None
    type: Literal["gol_valido"] = field(default="gol_valido", init=False)


@dataclass(kw_only=True)
class InvalidGoalEvent(Event):
    team: Team
    reason: InvalidGoalReason
    infracting_robot: Robot
    type: Literal["gol_invalido"] = field(default="gol_invalido", init=False)


@dataclass(kw_only=True)
class RemoveRobotEvent(Event):
    robot: Robot
    reason: RobotExpulsionReason
    frames_penalization: int
    penalization_s: Optional[float] = None
    type: Literal["sacar_robot"] = field(default="sacar_robot", init=False)


@dataclass(kw_only=True)
class PanicEvent(Event):
    reason: PanicReason
    frames_duration: int
    duration_s: Optional[float] = None
    type: Literal["panic"] = field(default="panic", init=False)


FrameEvent = Union[
    CollisionEvent,
    PossessionEvent,
    PassEvent,
    OffCourtEvent,
    RobotStoppedEvent,
    BallRepositionEvent,
    RobotRepositionEvent,
    GoalEvent,
    InvalidGoalEvent,
    RemoveRobotEvent,
    PanicEvent,
]


@dataclass
class FrameEvents:
    frame: int = 0
    timestamp_s: float = 0.0
    eventos: List[FrameEvent] = field(default_factory=list)
