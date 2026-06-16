from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot
from typing import Dict, Iterable, List, Optional, Set, Tuple

import domain.entities as domain_entities


try:
    Team = domain_entities.Team
except AttributeError:

    @dataclass
    class Team:
        name: str
        color: str
        score: int = 0
        robots: List["Robot"] = field(default_factory=list)

    domain_entities.Team = Team


from domain.entities import Ball, FrameResult, Robot
from domain.events import (
    BallRepositionEvent,
    CollisionEvent,
    FrameEvents,
    GoalEvent,
    InvalidGoalEvent,
    OffCourtEvent,
    PanicEvent,
    PassEvent,
    PossessionEvent,
    RemoveRobotEvent,
    RobotRepositionEvent,
    RobotStoppedEvent,
)
from infra.event_bus import EventBus


COURT_WIDTH_CM = 243.0
COURT_HEIGHT_CM = 182.0
GOAL_Y_MIN_CM = 61.0
GOAL_Y_MAX_CM = 121.0
ROBOT_RADIUS_CM = 11.0


class EventDetector:
    def __init__(self, event_bus: Optional[EventBus] = None, *, default_fps: float = 30.0) -> None:
        self._event_bus = event_bus
        self._default_fps = default_fps
        self._prev: Optional[FrameResult] = None
        self._last_valid_positions: Dict[str, Tuple[float, float]] = {}
        self._stopped_frames: Dict[str, int] = {}
        self._stopped_emitted: Set[str] = set()
        self._prev_penalized: Dict[str, bool] = {}
        self._last_possession_robot: Optional[Robot] = None
        self._pending_pass_from: Optional[Robot] = None

        if self._event_bus is not None:
            self._event_bus.subscribe("frame_result", self._on_frame_result)

    def detect(self, frame_result: FrameResult) -> FrameEvents:
        frame = int(frame_result.frame_id)
        timestamp_s = self._timestamp_s(frame_result)
        events = FrameEvents(frame=frame, timestamp_s=timestamp_s, eventos=[])

        valid_robots = [robot for robot in frame_result.robots if self._position(robot) is not None]
        ball_position = self._position(frame_result.ball) if frame_result.ball is not None else None

        events.eventos.extend(self._detect_panic(frame_result, frame, timestamp_s))
        events.eventos.extend(self._detect_collisions(valid_robots, frame, timestamp_s))
        events.eventos.extend(self._detect_off_court(frame_result, valid_robots, ball_position, frame, timestamp_s))
        events.eventos.extend(self._detect_stopped_robots(frame_result.robots, frame, timestamp_s))
        events.eventos.extend(self._detect_remove_robot(frame_result.robots, frame, timestamp_s))
        events.eventos.extend(self._detect_repositions(frame_result, frame, timestamp_s))

        possession_event = self._detect_possession(valid_robots, frame_result.ball, ball_position, frame, timestamp_s)
        if possession_event is not None:
            events.eventos.append(possession_event)

        pass_event = self._detect_pass(possession_event, frame, timestamp_s)
        if pass_event is not None:
            events.eventos.append(pass_event)

        goal_event = self._detect_goal(frame_result.ball, ball_position, frame, timestamp_s)
        if goal_event is not None:
            invalid_goal_event = self._detect_invalid_goal(goal_event, frame_result.robots, frame, timestamp_s)
            events.eventos.append(invalid_goal_event if invalid_goal_event is not None else goal_event)

        self._update_last_valid_positions(frame_result.robots, frame_result.ball)
        self._prev = frame_result

        if self._event_bus is not None:
            self._event_bus.publish("frame_events", events)
        return events

    def _on_frame_result(self, frame_result: FrameResult) -> None:
        self.detect(frame_result)

    def _detect_panic(self, frame_result: FrameResult, frame: int, timestamp_s: float) -> List[PanicEvent]:
        if self._prev is None:
            return []

        prev_ids = self._tracker_ids(self._prev)
        current_ids = self._tracker_ids(frame_result)
        changes = len(current_ids - prev_ids) + len(prev_ids - current_ids)
        if changes > 2:
            return [
                PanicEvent(
                    frame=frame,
                    timestamp_s=timestamp_s,
                    severity="system",
                    reason="ruido_visual",
                    frames_duration=1,
                )
            ]
        return []

    def _detect_collisions(self, robots: List[Robot], frame: int, timestamp_s: float) -> List[CollisionEvent]:
        events: List[CollisionEvent] = []
        for index, robot_a in enumerate(robots):
            pos_a = self._position(robot_a)
            if pos_a is None:
                continue
            for robot_b in robots[index + 1 :]:
                pos_b = self._position(robot_b)
                if pos_b is None:
                    continue
                if self._distance(pos_a, pos_b) < 20.0:
                    midpoint = ((pos_a[0] + pos_b[0]) / 2.0, (pos_a[1] + pos_b[1]) / 2.0)
                    events.append(
                        CollisionEvent(
                            frame=frame,
                            timestamp_s=timestamp_s,
                            position_cm=midpoint,
                            severity="warning",
                            robots=[robot_a, robot_b],
                        )
                    )
        return events

    def _detect_possession(
        self,
        robots: List[Robot],
        ball: Optional[Ball],
        ball_position: Optional[Tuple[float, float]],
        frame: int,
        timestamp_s: float,
    ) -> Optional[PossessionEvent]:
        if ball is None or ball_position is None:
            self._last_possession_robot = None
            return None

        closest_robot: Optional[Robot] = None
        closest_distance = 25.0
        for robot in robots:
            if not self._is_ally(robot):
                continue
            robot_position = self._position(robot)
            if robot_position is None:
                continue
            distance = self._distance(robot_position, ball_position)
            if distance < closest_distance:
                closest_robot = robot
                closest_distance = distance

        previous_possession = self._last_possession_robot
        if previous_possession is not None and (
            closest_robot is None or previous_possession.id != closest_robot.id
        ):
            self._pending_pass_from = previous_possession
        self._last_possession_robot = closest_robot

        if closest_robot is None:
            return None
        return PossessionEvent(
            frame=frame,
            timestamp_s=timestamp_s,
            robot=closest_robot,
            team=closest_robot.team_id,
            distance_cm=closest_distance,
        )

    def _detect_pass(
        self,
        possession_event: Optional[PossessionEvent],
        frame: int,
        timestamp_s: float,
    ) -> Optional[PassEvent]:
        if possession_event is None or self._pending_pass_from is None:
            return None

        from_robot = self._pending_pass_from
        to_robot = possession_event.robot
        if from_robot.id == to_robot.id or not self._is_ally(to_robot):
            self._pending_pass_from = None
            return None

        from_position = self._position(from_robot)
        to_position = self._position(to_robot)
        self._pending_pass_from = None
        if from_position is None or to_position is None:
            return None

        return PassEvent(
            frame=frame,
            timestamp_s=timestamp_s,
            from_robot_id=from_robot.id,
            to_robot_id=to_robot.id,
            from_team=from_robot.team_id,
            to_team=to_robot.team_id,
            distance_cm=self._distance(from_position, to_position),
        )

    def _detect_off_court(
        self,
        frame_result: FrameResult,
        robots: List[Robot],
        ball_position: Optional[Tuple[float, float]],
        frame: int,
        timestamp_s: float,
    ) -> List[OffCourtEvent]:
        events: List[OffCourtEvent] = []
        if frame_result.ball is not None and ball_position is not None and not self._inside_court(ball_position):
            last_position = self._last_valid_positions.get(frame_result.ball.id)
            if last_position is not None:
                events.append(
                    OffCourtEvent(
                        frame=frame,
                        timestamp_s=timestamp_s,
                        severity="warning",
                        object_type="balon",
                        object_id=frame_result.ball.id,
                        last_position_cm=last_position,
                    )
                )

        for robot in robots:
            position = self._position(robot)
            if position is None or self._inside_court(position):
                continue
            last_position = self._last_valid_positions.get(robot.id)
            if last_position is not None:
                events.append(
                    OffCourtEvent(
                        frame=frame,
                        timestamp_s=timestamp_s,
                        severity="warning",
                        object_type="robot",
                        object_id=robot.id,
                        last_position_cm=last_position,
                    )
                )
        return events

    def _detect_stopped_robots(self, robots: List[Robot], frame: int, timestamp_s: float) -> List[RobotStoppedEvent]:
        events: List[RobotStoppedEvent] = []
        for robot in robots:
            if robot.speed < 2.0:
                frames = self._stopped_frames.get(robot.id, 0) + 1
                self._stopped_frames[robot.id] = frames
                if frames > 30 and robot.id not in self._stopped_emitted:
                    self._stopped_emitted.add(robot.id)
                    events.append(
                        RobotStoppedEvent(
                            frame=frame,
                            timestamp_s=timestamp_s,
                            robot=robot,
                            frames_duration=frames,
                            duration_s=frames / self._default_fps if self._default_fps else None,
                        )
                    )
            else:
                self._stopped_frames[robot.id] = 0
                self._stopped_emitted.discard(robot.id)
        return events

    def _detect_goal(
        self,
        ball: Optional[Ball],
        ball_position: Optional[Tuple[float, float]],
        frame: int,
        timestamp_s: float,
    ) -> Optional[GoalEvent]:
        if self._prev is None or ball is None or ball_position is None or self._prev.ball is None:
            return None
        previous_position = self._position(self._prev.ball)
        if previous_position is None or not (GOAL_Y_MIN_CM <= ball_position[1] <= GOAL_Y_MAX_CM):
            return None

        if previous_position[0] < COURT_WIDTH_CM <= ball_position[0]:
            return GoalEvent(
                frame=frame,
                timestamp_s=timestamp_s,
                position_cm=ball_position,
                severity="success",
                team="allies",
                velocity_cm_s=ball.speed_cm_s,
            )
        if previous_position[0] > 0.0 >= ball_position[0]:
            return GoalEvent(
                frame=frame,
                timestamp_s=timestamp_s,
                position_cm=ball_position,
                severity="success",
                team="rivals",
                velocity_cm_s=ball.speed_cm_s,
            )
        return None

    def _detect_invalid_goal(
        self,
        goal_event: GoalEvent,
        robots: List[Robot],
        frame: int,
        timestamp_s: float,
    ) -> Optional[InvalidGoalEvent]:
        area = "rival" if self._team_name(goal_event.team) in {"allies", "ally", "azul", "blue"} else "ally"
        for robot in robots:
            position = self._position(robot)
            if position is not None and self._robot_inside_penalty_area(position, area):
                return InvalidGoalEvent(
                    frame=frame,
                    timestamp_s=timestamp_s,
                    position_cm=goal_event.position_cm,
                    severity="danger",
                    team=goal_event.team,
                    reason="robot_en_area_penalti",
                    infracting_robot=robot,
                )
        return None

    def _detect_remove_robot(self, robots: List[Robot], frame: int, timestamp_s: float) -> List[RemoveRobotEvent]:
        events: List[RemoveRobotEvent] = []
        for robot in robots:
            was_penalized = self._prev_penalized.get(robot.id, False)
            if robot.is_penalized and not was_penalized:
                reason = "ingreso_area_penalti" if self._is_robot_in_any_penalty_area(robot) else "toco_pared_cancha"
                events.append(
                    RemoveRobotEvent(
                        frame=frame,
                        timestamp_s=timestamp_s,
                        severity="danger",
                        robot=robot,
                        reason=reason,
                        frames_penalization=robot.penalization_frames_left,
                        penalization_s=robot.penalization_frames_left / self._default_fps if self._default_fps else None,
                    )
                )
            self._prev_penalized[robot.id] = robot.is_penalized
        return events

    def _detect_repositions(
        self,
        frame_result: FrameResult,
        frame: int,
        timestamp_s: float,
    ) -> List[BallRepositionEvent | RobotRepositionEvent]:
        events: List[BallRepositionEvent | RobotRepositionEvent] = []
        robot_by_id = {robot.id: robot for robot in frame_result.robots}

        for reposition in getattr(frame_result, "repositions", []) or []:
            event_type = str(reposition.get("type", "")).lower()
            object_type = str(reposition.get("object_type", reposition.get("object", ""))).lower()

            if event_type == "reposicion_balon" or object_type in {"ball", "balon"}:
                to_position = self._tuple_or_none(
                    reposition.get("to_position_cm", reposition.get("target_position_cm", reposition.get("position_cm")))
                )
                if to_position is None:
                    continue

                from_position = self._tuple_or_none(reposition.get("from_position_cm"))
                if from_position is None and frame_result.ball is not None:
                    from_position = self._last_valid_positions.get(frame_result.ball.id)

                events.append(
                    BallRepositionEvent(
                        frame=frame,
                        timestamp_s=timestamp_s,
                        position_cm=to_position,
                        severity="warning",
                        reason=reposition.get("reason", "falta_de_progreso"),
                        from_position_cm=from_position,
                        to_position_cm=to_position,
                    )
                )
                continue

            if event_type == "reposicion_robot" or object_type == "robot":
                robot_id = str(reposition.get("robot_id", reposition.get("object_id", reposition.get("robot", ""))))
                robot = robot_by_id.get(robot_id)
                target_position = self._tuple_or_none(
                    reposition.get("target_position_cm", reposition.get("to_position_cm", reposition.get("position_cm")))
                )
                if robot is None or target_position is None:
                    continue

                from_position = self._tuple_or_none(reposition.get("from_position_cm"))
                if from_position is None:
                    from_position = self._last_valid_positions.get(robot.id)

                events.append(
                    RobotRepositionEvent(
                        frame=frame,
                        timestamp_s=timestamp_s,
                        position_cm=target_position,
                        severity="warning",
                        robot=robot,
                        from_position_cm=from_position,
                        target_position_cm=target_position,
                    )
                )

        return events

    def _update_last_valid_positions(self, robots: Iterable[Robot], ball: Optional[Ball]) -> None:
        for robot in robots:
            position = self._position(robot)
            if position is not None and self._inside_court(position):
                self._last_valid_positions[robot.id] = position
        if ball is not None:
            position = self._position(ball)
            if position is not None and self._inside_court(position):
                self._last_valid_positions[ball.id] = position

    def _is_robot_in_any_penalty_area(self, robot: Robot) -> bool:
        position = self._position(robot)
        return position is not None and (
            self._robot_inside_penalty_area(position, "ally") or self._robot_inside_penalty_area(position, "rival")
        )

    @staticmethod
    def _position(obj: object) -> Optional[Tuple[float, float]]:
        if obj is None:
            return None
        metric = getattr(obj, "position_metric", None)
        if metric is not None:
            return (float(metric.x), float(metric.y))
        return None

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _inside_court(position: Tuple[float, float]) -> bool:
        return 0.0 <= position[0] <= COURT_WIDTH_CM and 0.0 <= position[1] <= COURT_HEIGHT_CM

    @staticmethod
    def _robot_inside_penalty_area(position: Tuple[float, float], area: str) -> bool:
        x, y = position
        if area == "ally":
            return x - ROBOT_RADIUS_CM >= 0.0 and x + ROBOT_RADIUS_CM <= 25.0 and y - ROBOT_RADIUS_CM >= 51.0 and y + ROBOT_RADIUS_CM <= 131.0
        return x - ROBOT_RADIUS_CM >= 218.0 and x + ROBOT_RADIUS_CM <= 243.0 and y - ROBOT_RADIUS_CM >= 51.0 and y + ROBOT_RADIUS_CM <= 131.0

    @staticmethod
    def _tracker_ids(frame_result: FrameResult) -> Set[int]:
        ids = {robot.tracker_id for robot in frame_result.robots if robot.tracker_id is not None}
        if frame_result.ball is not None and frame_result.ball.tracker_id is not None:
            ids.add(frame_result.ball.tracker_id)
        return ids

    @staticmethod
    def _is_ally(robot: Robot) -> bool:
        return robot.team_id.lower() in {"allies", "ally", "azul", "blue"}

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

    def _timestamp_s(self, frame_result: FrameResult) -> float:
        timestamp_s = getattr(frame_result, "timestamp_s", None)
        if timestamp_s is not None:
            return float(timestamp_s)
        if self._default_fps:
            return float(frame_result.frame_id) / self._default_fps
        return 0.0
