from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FieldGeometry:
    outer_width_cm: float = 243.0
    outer_height_cm: float = 182.0
    boundary_inset_cm: float = 12.0
    center_circle_diameter_cm: float = 60.0
    center_spot_radius_cm: float = 2.0
    penalty_box_depth_cm: float = 25.0
    penalty_box_height_cm: float = 80.0
    penalty_box_corner_radius_cm: float = 6.0
    goal_depth_cm: float = 10.0
    goal_height_cm: float = 60.0
    robot_radius_cm: float = 11.0

    @property
    def inner_width_cm(self) -> float:
        return self.outer_width_cm - 2.0 * self.boundary_inset_cm

    @property
    def inner_height_cm(self) -> float:
        return self.outer_height_cm - 2.0 * self.boundary_inset_cm

    @property
    def center_x_cm(self) -> float:
        return self.outer_width_cm / 2.0

    @property
    def center_y_cm(self) -> float:
        return self.outer_height_cm / 2.0

    @property
    def goal_y_min_cm(self) -> float:
        return self.center_y_cm - self.goal_height_cm / 2.0

    @property
    def goal_y_max_cm(self) -> float:
        return self.center_y_cm + self.goal_height_cm / 2.0

    @property
    def left_goal_entry_x_cm(self) -> float:
        return self.boundary_inset_cm

    @property
    def right_goal_entry_x_cm(self) -> float:
        return self.outer_width_cm - self.boundary_inset_cm


FIELD_GEOMETRY = FieldGeometry()


def inside_court(position: Tuple[float, float], geometry: FieldGeometry = FIELD_GEOMETRY) -> bool:
    x, y = position
    return 0.0 <= x <= geometry.outer_width_cm and 0.0 <= y <= geometry.outer_height_cm


def inside_goal_mouth_y(y_cm: float, geometry: FieldGeometry = FIELD_GEOMETRY) -> bool:
    return geometry.goal_y_min_cm <= y_cm <= geometry.goal_y_max_cm


def crossed_right_goal_entry(
    previous_x_cm: float,
    current_x_cm: float,
    geometry: FieldGeometry = FIELD_GEOMETRY,
) -> bool:
    return previous_x_cm < geometry.right_goal_entry_x_cm <= current_x_cm


def crossed_left_goal_entry(
    previous_x_cm: float,
    current_x_cm: float,
    geometry: FieldGeometry = FIELD_GEOMETRY,
) -> bool:
    return previous_x_cm > geometry.left_goal_entry_x_cm >= current_x_cm


def robot_inside_penalty_area(
    position: Tuple[float, float],
    area: str,
    geometry: FieldGeometry = FIELD_GEOMETRY,
) -> bool:
    x, y = position
    r = geometry.robot_radius_cm
    if area == "ally":
        return (
            x - r >= 0.0
            and x + r <= geometry.penalty_box_depth_cm
            and y - r >= geometry.center_y_cm - geometry.penalty_box_height_cm / 2.0
            and y + r <= geometry.center_y_cm + geometry.penalty_box_height_cm / 2.0
        )
    return (
        x - r >= geometry.outer_width_cm - geometry.penalty_box_depth_cm
        and x + r <= geometry.outer_width_cm
        and y - r >= geometry.center_y_cm - geometry.penalty_box_height_cm / 2.0
        and y + r <= geometry.center_y_cm + geometry.penalty_box_height_cm / 2.0
    )
