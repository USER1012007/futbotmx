from copy import deepcopy
from typing import Dict, Tuple

from domain.field import FIELD_GEOMETRY

MAX_BALL_SPEED_CM_S = 600.0
MAX_ROBOTS_PER_TEAM = 2
TEAM_SIDE_MARGIN_CM = 10.0
SLOT_REID_DISTANCE_CM = 55.0
SLOT_REID_DISTANCE_PX = 260.0
TEAM_SLOTS = {
    "allies": ("A1", "A2"),
    "rivals": ("R1", "R2"),
}


class TeamAssigner:
    def __init__(self):
        self._tracker_identity: Dict[int, Tuple[str, str]] = {}
        self._slot_positions: Dict[str, tuple[float, float, bool]] = {}
        self._frame_width: int | None = None

    def assign(self, result, frame=None) -> None:
        self._frame_width = frame.shape[1] if frame is not None else None
        if not result.robots:
            if result.ball is not None:
                result.ball.id = "ball"
            return

        team_groups = self._group_by_side(result.robots)
        assigned = []

        for team_id in ("allies", "rivals"):
            team_robots = self._limit_team(team_groups[team_id], team_id)
            used_ids: set[str] = set()
            for robot in team_robots:
                tracker_id = int(robot.tracker_id)
                robot_id = self._identity_for_robot(robot, team_id, used_ids)
                if robot_id is None:
                    continue

                robot.id = robot_id
                robot.team_id = team_id
                used_ids.add(robot_id)
                assigned.append(robot)
                self._tracker_identity[tracker_id] = (robot_id, team_id)
                self._slot_positions[robot_id] = self._robot_position(robot)

        result.robots = assigned

        if result.ball is not None:
            result.ball.id = "ball"

    def _group_by_side(self, robots) -> dict[str, list]:
        with_metric = [robot for robot in robots if getattr(robot, "position_metric", None) is not None]
        if len(with_metric) >= MAX_ROBOTS_PER_TEAM * 2:
            ordered = sorted(with_metric, key=lambda robot: robot.position_metric.x)
            return {
                "allies": ordered[:MAX_ROBOTS_PER_TEAM],
                "rivals": ordered[-MAX_ROBOTS_PER_TEAM:],
            }
        if len(robots) >= MAX_ROBOTS_PER_TEAM * 2:
            ordered = sorted(robots, key=lambda robot: robot.position_pixel.x)
            return {
                "allies": ordered[:MAX_ROBOTS_PER_TEAM],
                "rivals": ordered[-MAX_ROBOTS_PER_TEAM:],
            }

        groups = {"allies": [], "rivals": []}
        for robot in robots:
            groups[self._side_for_robot(robot)].append(robot)

        overflow = []
        for team_id in ("allies", "rivals"):
            if len(groups[team_id]) > MAX_ROBOTS_PER_TEAM:
                groups[team_id].sort(key=lambda robot: self._side_score(robot, team_id), reverse=True)
                overflow.extend(groups[team_id][MAX_ROBOTS_PER_TEAM:])
                groups[team_id] = groups[team_id][:MAX_ROBOTS_PER_TEAM]

        for robot in overflow:
            other_team = "rivals" if self._side_for_robot(robot) == "allies" else "allies"
            if len(groups[other_team]) < MAX_ROBOTS_PER_TEAM:
                groups[other_team].append(robot)

        return groups

    def _side_for_robot(self, robot) -> str:
        tracker_id = int(robot.tracker_id)
        previous = self._tracker_identity.get(tracker_id)
        x = self._x_position(robot)
        if x is None:
            return previous[1] if previous is not None else "allies"

        center_x = FIELD_GEOMETRY.center_x_cm if self._has_metric(robot) else self._pixel_center_x()
        if center_x is None:
            return previous[1] if previous is not None else "allies"

        if previous is not None:
            if previous[1] == "allies" and x <= center_x + TEAM_SIDE_MARGIN_CM:
                return "allies"
            if previous[1] == "rivals" and x >= center_x - TEAM_SIDE_MARGIN_CM:
                return "rivals"

        return "allies" if x <= center_x else "rivals"

    def _identity_for_robot(
        self,
        robot,
        team_id: str,
        used_ids: set[str],
    ) -> str | None:
        tracker_id = int(robot.tracker_id)
        previous = self._tracker_identity.get(tracker_id)
        if previous is not None and previous[1] == team_id and previous[0] not in used_ids:
            return previous[0]

        nearest_slot = self._nearest_free_slot(robot, team_id, used_ids)
        if nearest_slot is not None:
            return nearest_slot

        for robot_id in TEAM_SLOTS[team_id]:
            if robot_id not in used_ids:
                return robot_id
        return None

    def _nearest_free_slot(
        self,
        robot,
        team_id: str,
        used_ids: set[str],
    ) -> str | None:
        current = self._robot_position(robot)
        best_id = None
        best_distance = float("inf")
        for robot_id in TEAM_SLOTS[team_id]:
            if robot_id in used_ids or robot_id not in self._slot_positions:
                continue
            distance = self._position_distance(current, self._slot_positions[robot_id])
            if distance < best_distance:
                best_id = robot_id
                best_distance = distance

        if best_id is None:
            return None

        threshold = SLOT_REID_DISTANCE_CM if current[2] else SLOT_REID_DISTANCE_PX
        return best_id if best_distance <= threshold else None

    def _limit_team(self, robots, team_id: str) -> list:
        if len(robots) <= MAX_ROBOTS_PER_TEAM:
            return self._order_team_slots(robots)

        robots = sorted(robots, key=lambda robot: self._side_score(robot, team_id), reverse=True)
        return self._order_team_slots(robots[:MAX_ROBOTS_PER_TEAM])

    @staticmethod
    def _order_team_slots(robots) -> list:
        return sorted(robots, key=lambda robot: TeamAssigner._y_position(robot))

    def _side_score(self, robot, team_id: str) -> float:
        x = self._x_position(robot)
        if x is None:
            return 0.0

        if self._has_metric(robot):
            return FIELD_GEOMETRY.center_x_cm - x if team_id == "allies" else x - FIELD_GEOMETRY.center_x_cm

        return -x if team_id == "allies" else x

    @staticmethod
    def _robot_position(robot) -> tuple[float, float, bool]:
        if getattr(robot, "position_metric", None) is not None:
            return robot.position_metric.x, robot.position_metric.y, True
        return robot.position_pixel.x, robot.position_pixel.y, False

    @staticmethod
    def _position_distance(
        current: tuple[float, float, bool],
        previous: tuple[float, float, bool],
    ) -> float:
        if current[2] != previous[2]:
            return float("inf")
        return ((current[0] - previous[0]) ** 2 + (current[1] - previous[1]) ** 2) ** 0.5

    @staticmethod
    def _has_metric(robot) -> bool:
        return getattr(robot, "position_metric", None) is not None

    @staticmethod
    def _x_position(robot) -> float | None:
        if getattr(robot, "position_metric", None) is not None:
            return robot.position_metric.x
        if getattr(robot, "position_pixel", None) is not None:
            return robot.position_pixel.x
        return None

    @staticmethod
    def _y_position(robot) -> float:
        if getattr(robot, "position_metric", None) is not None:
            return robot.position_metric.y
        return robot.position_pixel.y

    def _pixel_center_x(self) -> float | None:
        if self._frame_width is None:
            return None
        return self._frame_width / 2.0


class BallMetricStabilizer:
    def __init__(self, fps: float):
        self._fps = fps
        self._last_valid_ball = None
        self._last_valid_ball_frame: int | None = None

    def stabilize(self, result) -> None:
        ball = result.ball
        if ball is None or ball.position_metric is None:
            if self._last_valid_ball is not None and self._last_valid_ball_frame is not None:
                missing_frames = result.frame_id - self._last_valid_ball_frame
                if missing_frames <= 8:
                    result.ball = deepcopy(self._last_valid_ball)
            return

        if self._last_valid_ball is not None and self._last_valid_ball.position_metric is not None:
            dt_frames = max(1, result.frame_id - (self._last_valid_ball_frame or result.frame_id))
            dt = dt_frames / self._fps if self._fps else dt_frames / 30.0
            prev = self._last_valid_ball.position_metric
            curr = ball.position_metric
            distance = ((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2) ** 0.5
            speed_cm_s = distance / max(dt, 1e-6)
            if speed_cm_s > MAX_BALL_SPEED_CM_S:
                result.ball = deepcopy(self._last_valid_ball)
                return

        self._last_valid_ball = deepcopy(result.ball)
        self._last_valid_ball_frame = result.frame_id
