from __future__ import annotations

from visualization.common.types import Color


BACKGROUND: Color = (18, 22, 28)
PANEL: Color = (31, 38, 48)
PANEL_ALT: Color = (40, 48, 59)
WHITE: Color = (245, 248, 252)
MUTED: Color = (160, 170, 182)
ALLIES: Color = (216, 180, 0)
RIVALS: Color = (60, 35, 239)
ACCENT: Color = (0, 210, 255)
BORDER: Color = (58, 68, 82)

DASHBOARD_HISTORY_MAX_EVENTS = 8

_HEADER_H_RATIO = 0.16
_POSSESSION_H_RATIO = 0.14
_DISTANCE_H_RATIO = 0.20
_EVENT_SUMMARY_H_RATIO = 0.18
_MARGIN_RATIO = 0.025
_GAP_RATIO = 0.015


class DashboardLayout:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

        self.margin = max(6, int(min(width, height) * _MARGIN_RATIO))
        self.gap = max(4, int(height * _GAP_RATIO))
        self.content_w = max(1, width - 2 * self.margin)

        self.header_h = max(36, int(height * _HEADER_H_RATIO))
        self.possession_h = max(36, int(height * _POSSESSION_H_RATIO))
        self.distance_h = max(36, int(height * _DISTANCE_H_RATIO))
        self.event_summary_h = max(36, int(height * _EVENT_SUMMARY_H_RATIO))

        used = (
            self.margin
            + self.header_h + self.gap
            + self.possession_h + self.gap
            + self.distance_h + self.gap
            + self.event_summary_h + self.gap
            + self.margin
        )
        self.story_h = max(36, height - used)

        self.header_y = self.margin
        self.possession_y = self.header_y + self.header_h + self.gap
        self.distance_y = self.possession_y + self.possession_h + self.gap
        self.event_summary_y = self.distance_y + self.distance_h + self.gap
        self.story_y = self.event_summary_y + self.event_summary_h + self.gap

        base_w = width / 520.0
        base_h = height / 1080.0
        base = max(0.85, min(1.35, (base_w * 0.75) + (base_h * 0.25)))

        self.scale_xs = 0.55 * base
        self.scale_sm = 0.70 * base
        self.scale_md = 0.82 * base
        self.scale_lg = 1.08 * base
        self.scale_xl = 1.55 * base

        self.thick_sm = 1
        self.thick_md = max(1, min(2, int(round(1.4 * base))))
        self.thick_lg = max(2, min(4, int(round(2.2 * base))))
        self.bar_h = max(10, int(height * 0.028))
        self.row_h_min = max(22, int(height * 0.045))
