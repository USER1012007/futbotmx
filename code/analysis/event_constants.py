from typing import Dict

EVENT_COOLDOWN_FRAMES: Dict[str, int] = {
    "pase": 15,
    "colision": 30,
    "fuera_de_cancha": 30,
    "gol_valido": 90,
    "gol_invalido": 90,
    "robot_detenido": 90,
    "sacar_robot": 90,
    "panic": 45,
    "reposicion_balon": 45,
    "reposicion_robot": 45,
}

GOAL_CONTACT_DISTANCE_CM = 38.0
GOAL_CONTACT_MEMORY_FRAMES = 90
REQUIRE_RECENT_BALL_CONTACT_FOR_GOAL = True
RIGHT_GOAL_FALLBACK_TEAM = "allies"
LEFT_GOAL_FALLBACK_TEAM = "rivals"
