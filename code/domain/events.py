from dataclasses import dataclass
from typing import List, Tuple, Literal, Union

@dataclass
class Event:
    pass

@dataclass
class ColisionEvent(Event):
    tipo: Literal["colision"] 
    robots: List[Robot] 
    pos: Tuple[float, float] 

@dataclass
class PosesionEvent(Event):
    tipo: Literal["posesion"]
    robot: Robot 
    distancia_cm: float 

@dataclass
class TiroAGolEvent(Event):
    tipo: Literal["tiro_a_gol"]
    equipo: Literal["aliado", "rival"]
    velocidad_cm_s: float
    porteria: Literal["propia", "rival"]

@dataclass
class PaseEvent(Event):
    tipo: Literal["pase"]
    de:  Robot.id
    a: Robot.id
    distancia_cm: float

@dataclass
class FueraDeCanchaEvent(Event):
    tipo: Literal["fuera_de_cancha"]
    objeto: Literal[Ball, Robot] 
    pos_ultima: Tuple[float, float] 

@dataclass
class RobotDetenidoEvent(Event):
    tipo: Literal["robot_detenido"]
    robot: Robot.id
    duracion_frames: int 

@dataclass
class PanicEvent(Event):
    tipo: Literal["panic"]
    duracion_frames: int 

@dataclass
class FrameEvents:
    eventos: List[Union[ColisionEvent,
                        PosesionEvent,
                        TiroAGolEvent,
                        PaseEvent,
                        FueraDeCanchaEvent,
                        RobotDetenidoEvent,
                        PanicEvent]]

