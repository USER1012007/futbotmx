from dataclasses import dataclass
from typing import List, Tuple, Literal, Union
from domain.entities import Robot, Ball, Team

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
class ShotOnGoalEvent(Event):
    type: Literal["tiro a gol"]
    team: Team
    velocity_cm_s: float
    goal: str

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
class PanicEvent(Event):
    type: Literal["panic"]
    frames_duration: int 

@dataclass
class FrameEvents:
    eventos: List[Union[CollisionEvent, PossessionEvent, ShotOnGoalEvent, PassEvent, OffCourtEvent, RobotStoppedEvent, PanicEvent]]

