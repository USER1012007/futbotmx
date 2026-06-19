from __future__ import annotations

from math import hypot
from typing import Dict, Optional, Tuple


from domain.entities import FrameResult, Robot
from domain.events import (
    CollisionEvent,
    FrameEvents,
    GoalEvent,
    InvalidGoalEvent,
    PassEvent,
    PossessionEvent,
    RemoveRobotEvent,
    RobotStoppedEvent,
)
from domain.stats import PossessionPct, Score, Statistics
from infra.event_bus import EventBus

ROBOT_DISTANCE_DEADBAND_CM = 4.0
ROBOT_MAX_DISTANCE_SPEED_CM_S = 250.0


class StatsEngine:
    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self._event_bus = event_bus
        self._score = Score()
        self._frames_ally_possession = 0
        self._frames_rival_possession = 0
        self._distance_cm: Dict[str, float] = {"allies": 0.0, "rivals": 0.0}
        self._prev_positions: Dict[str, Tuple[float, float]] = {}
        self._prev_position_frames: Dict[str, int] = {}
        self._distance_anchor_positions: Dict[str, Tuple[float, float]] = {}
        self._event_counts: Dict[str, int] = {}
        self._passes_attempted = 0
        self._passes_successful = 0
        self._collisions = 0
        self._invalid_goals = 0
        self._penalizations = 0
        self._stopped_frames_by_robot: Dict[str, int] = {}
        self._possession_frames_by_robot: Dict[str, int] = {}
        self._speed_sum_cm_s: Dict[str, float] = {}
        self._speed_samples: Dict[str, int] = {}
        self._max_speed_cm_s: Dict[str, float] = {}
        self._latest_frame_events = FrameEvents()

        if self._event_bus is not None:
            self._event_bus.subscribe("frame_events", self._on_frame_events)
            self._event_bus.subscribe("frame_result", self._on_frame_result)

    def update(self, frame_result: FrameResult, frame_events: FrameEvents) -> Statistics:
        self._update_score(frame_events)
        self._update_event_stats(frame_events)
        self._update_possession(frame_result, frame_events)
        self._update_distances(frame_result)
        self._update_speeds(frame_result)
        return self._statistics()

    def _on_frame_events(self, frame_events: FrameEvents) -> None:
        self._latest_frame_events = frame_events

    def _on_frame_result(self, frame_result: FrameResult) -> None:
        statistics = self.update(frame_result, self._latest_frame_events)
        if self._event_bus is not None:
            self._event_bus.publish("statistics", statistics)
        self._latest_frame_events = FrameEvents()

    def _update_score(self, frame_events: FrameEvents) -> None:
        for event in frame_events.eventos:
            if not isinstance(event, GoalEvent):
                continue
            team_name = self._team_name(event.team)
            if team_name in {"allies", "ally", "azul", "blue"}:
                self._score.allies += 1
            elif team_name in {"rivals", "rival", "rojo", "red"}:
                self._score.rivals += 1

    def _update_event_stats(self, frame_events: FrameEvents) -> None:
        for event in frame_events.eventos:
            event_type = getattr(event, "type", event.__class__.__name__)
            self._event_counts[event_type] = self._event_counts.get(event_type, 0) + 1

            if isinstance(event, PassEvent):
                self._passes_attempted += 1
                if event.successful:
                    self._passes_successful += 1
            elif isinstance(event, CollisionEvent):
                self._collisions += 1
            elif isinstance(event, InvalidGoalEvent):
                self._invalid_goals += 1
            elif isinstance(event, RemoveRobotEvent):
                self._penalizations += 1
            elif isinstance(event, RobotStoppedEvent):
                robot_id = event.robot.id
                self._stopped_frames_by_robot[robot_id] = max(
                    self._stopped_frames_by_robot.get(robot_id, 0),
                    int(event.frames_duration),
                )

    def _update_possession(self, frame_result: FrameResult, frame_events: FrameEvents) -> None:
        has_ball = frame_result.ball is not None and getattr(frame_result.ball, "position_metric", None) is not None
        has_ally_possession = any(isinstance(event, PossessionEvent) for event in frame_events.eventos)
        if has_ally_possession:
            self._frames_ally_possession += 1
            for event in frame_events.eventos:
                if isinstance(event, PossessionEvent):
                    robot_id = event.robot.id
                    self._possession_frames_by_robot[robot_id] = self._possession_frames_by_robot.get(robot_id, 0) + 1
        elif has_ball:
            self._frames_rival_possession += 1

    def _update_distances(self, frame_result: FrameResult) -> None:
        current_positions: Dict[str, Tuple[float, float]] = {}
        current_frames: Dict[str, int] = {}
        for robot in frame_result.robots:
            position = self._position(robot)
            if position is None:
                continue

            previous_position = self._prev_positions.get(robot.id)
            if previous_position is None:
                current_positions[robot.id] = position
                current_frames[robot.id] = frame_result.frame_id
                self._distance_cm.setdefault(robot.id, 0.0)
                self._distance_anchor_positions[robot.id] = position
                continue

            if self._is_implausible_distance_step(robot.id, position, frame_result.frame_id):
                current_positions[robot.id] = previous_position
                current_frames[robot.id] = self._prev_position_frames.get(robot.id, frame_result.frame_id)
                continue

            current_positions[robot.id] = position
            current_frames[robot.id] = frame_result.frame_id
            distance = self._distance_from_anchor(robot.id, position)
            if distance <= 0.0:
                continue

            self._distance_cm[robot.id] = self._distance_cm.get(robot.id, 0.0) + distance
            team_key = self._team_key(robot)
            if team_key is not None:
                self._distance_cm[team_key] = self._distance_cm.get(team_key, 0.0) + distance

        self._prev_positions = current_positions
        self._prev_position_frames = current_frames

    def _update_speeds(self, frame_result: FrameResult) -> None:
        for robot in frame_result.robots:
            speed = float(getattr(robot, "speed", getattr(robot, "speed_cm_s", 0.0)) or 0.0)
            self._speed_sum_cm_s[robot.id] = self._speed_sum_cm_s.get(robot.id, 0.0) + speed
            self._speed_samples[robot.id] = self._speed_samples.get(robot.id, 0) + 1
            self._max_speed_cm_s[robot.id] = max(self._max_speed_cm_s.get(robot.id, 0.0), speed)

    def _statistics(self) -> Statistics:
        total_possession_frames = self._frames_ally_possession + self._frames_rival_possession
        if total_possession_frames == 0:
            possession_pct = PossessionPct(allies=50.0, rivals=50.0)
        else:
            possession_pct = PossessionPct(
                allies=self._frames_ally_possession / total_possession_frames * 100.0,
                rivals=self._frames_rival_possession / total_possession_frames * 100.0,
            )
        avg_speed = {
            robot_id: self._speed_sum_cm_s[robot_id] / max(samples, 1)
            for robot_id, samples in self._speed_samples.items()
        }
        return Statistics(
            score=self._score,
            possession_pct=possession_pct,
            distance_cm=dict(self._distance_cm),
            event_counts=dict(self._event_counts),
            passes_attempted=self._passes_attempted,
            passes_successful=self._passes_successful,
            collisions=self._collisions,
            invalid_goals=self._invalid_goals,
            penalizations=self._penalizations,
            stopped_frames_by_robot=dict(self._stopped_frames_by_robot),
            possession_frames_by_robot=dict(self._possession_frames_by_robot),
            avg_speed_cm_s=avg_speed,
            max_speed_cm_s=dict(self._max_speed_cm_s),
        )

    @staticmethod
    def _position(robot: Robot) -> Optional[Tuple[float, float]]:
        metric = getattr(robot, "position_metric", None)
        if metric is not None:
            return (float(metric.x), float(metric.y))
        return None

    def _distance_from_anchor(self, robot_id: str, position: Tuple[float, float]) -> float:
        anchor = self._distance_anchor_positions.get(robot_id)
        if anchor is None:
            self._distance_anchor_positions[robot_id] = position
            return 0.0

        distance = hypot(position[0] - anchor[0], position[1] - anchor[1])
        if distance < ROBOT_DISTANCE_DEADBAND_CM:
            return 0.0

        self._distance_anchor_positions[robot_id] = position
        return distance

    def _is_implausible_distance_step(
        self,
        robot_id: str,
        position: Tuple[float, float],
        frame_id: int,
    ) -> bool:
        previous_position = self._prev_positions.get(robot_id)
        previous_frame = self._prev_position_frames.get(robot_id)
        if previous_position is None or previous_frame is None:
            return False

        frame_delta = max(1, frame_id - previous_frame)
        dt = frame_delta / 30.0
        distance = hypot(position[0] - previous_position[0], position[1] - previous_position[1])
        speed = distance / max(dt, 1e-6)
        return speed > ROBOT_MAX_DISTANCE_SPEED_CM_S

    @staticmethod
    def _team_key(robot: Robot) -> Optional[str]:
        team_id = robot.team_id.lower()
        if team_id in {"allies", "ally", "azul", "blue"}:
            return "allies"
        if team_id in {"rivals", "rival", "rojo", "red"}:
            return "rivals"
        return None

    @staticmethod
    def _team_name(team: object) -> str:
        if isinstance(team, str):
            return team.lower()
        return getattr(team, "name", str(team)).lower()
