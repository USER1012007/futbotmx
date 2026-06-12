from dataclasses import dataclass
from typing import List, Tuple, Literal, Union
from domain.entities import Robot, Ball, Team

BallRepositionReason = Literal["falta_de_progreso", "colision_area_penalti"]
RobotExpulsionReason = Literal["toco_pared_cancha", "ingreso_area_penalti"]
InvalidGoalReason = Literal["robot_en_area_penalti"]
PanicReason = Literal["ruido_visual"]

@dataclass
class Event:
    pass

@dataclass
class CollisionEvent(Event):
    type: Literal["colision"] 
    robots: List[Robot] 
    position: Tuple[float, float] 

@dataclass
class PossessionEvent(Event):
    type: Literal["posesion"]
    robot: Robot 
    distance_cm: float 

@dataclass
class PassEvent(Event):
    type: Literal["pase"]
    from_id: str
    to: str
    distance_cm: float

@dataclass
class OffCourtEvent(Event):
    type: Literal["fuera_de_cancha"]
    object: Literal["balon", "robot"]
    last_position: Tuple[float, float] 

@dataclass
class RobotStoppedEvent(Event):
    type: Literal["robot_detenido"]
    robot: Robot
    frames_duration: int 

@dataclass
class BallRepositionEvent(Event):
    type: Literal["reposicion_balon"]
    reason: BallRepositionReason
    position: Tuple[float, float]

@dataclass
class RobotRepositionEvent(Event):
    type: Literal["reposicion_robot"]
    robot: Robot
    target_position: Tuple[float, float]

@dataclass
class GoalEvent(Event):
    type: Literal["gol_valido"]
    team: Team
    velocity_cm_s: float
    position: Tuple[float, float]

@dataclass
class InvalidGoalEvent(Event):
    type: Literal["gol_invalido"]
    team: Team
    reason: InvalidGoalReason
    infracting_robot: Robot

@dataclass
class RemoveRobotEvent(Event):
    type: Literal["sacar_robot"]
    robot: Robot
    reason: RobotExpulsionReason
    frames_penalization: int 

@dataclass
class PanicEvent(Event):
    type: Literal["panic"]
    reason: PanicReason
    frames_duration: int 

@dataclass
class FrameEvents:
    eventos: List[Union[
        CollisionEvent, PossessionEvent, PassEvent, OffCourtEvent, 
        RobotStoppedEvent, BallRepositionEvent, RobotRepositionEvent, 
        GoalEvent, InvalidGoalEvent, RemoveRobotEvent, PanicEvent
    ]]
