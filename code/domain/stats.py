@dataclass
class Marcador:
    aliados: int = 0
    rivales: int = 0

@dataclass
class PosesionPct:
    aliados: float = 0.0
    rivales: float = 0.0
@dataclass
class DistanciaCm:
    distancia_aliadios: float = 0.0
    distancia_rivales: float = 0.0

@dataclass
class Estadisticas:
    posesion_pct: PosesionPct
    distancia_cm: List[DistanciaCm]

