"""Dashboard visual de FutBotMX.

Que hace: renderiza marcador, posesion, distancias y eventos en un panel BGR.
Flujo: recibe Statistics + FrameEvents, calcula layout y delega el dibujo de
secciones a modulos internos de visualization.
"""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Deque, Iterable, Mapping, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise ImportError("visualization.dashboard requires opencv-python and numpy.") from exc

from domain.stats import Statistics
from visualization.dashboard.events import count_events, event_key, get_event_counts
from visualization.dashboard.sections import draw_dashboard_sections
from visualization.dashboard.style import BACKGROUND, DASHBOARD_HISTORY_MAX_EVENTS, DashboardLayout

if TYPE_CHECKING:
    from domain.events import FrameEvents
    from infra.event_bus import EventBus


class DashboardRenderer:
    """Renderer de dashboard con soporte opcional de EventBus."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        *,
        statistics_event_type: str = "statistics",
        frame_events_event_type: str = "frame_events",
        video_frame_event_type: str = "video_frame",
        output_event_type: str = "dashboard",
        default_match_time_seconds: float = 0.0,
        dashboard_size: Optional[Tuple[int, int]] = None,
    ):
        self.output_event_type = output_event_type
        self.match_time_seconds = default_match_time_seconds
        self._event_bus = event_bus
        self._latest_stats: Optional[Statistics] = None
        self._latest_events: Optional[FrameEvents] = None
        self._dashboard_size = dashboard_size
        self._event_counts: Counter[str] = Counter()
        self._event_history: Deque[object] = deque(maxlen=DASHBOARD_HISTORY_MAX_EVENTS)
        self._seen_event_keys: set[tuple] = set()

        if event_bus is not None:
            event_bus.subscribe(statistics_event_type, self.on_statistics)
            event_bus.subscribe(frame_events_event_type, self.on_frame_events)
            event_bus.subscribe(video_frame_event_type, self.on_video_frame)

    def on_statistics(self, stats: Statistics) -> None:
        self._latest_stats = stats

    def on_frame_events(self, events: "FrameEvents") -> None:
        self._latest_events = events
        self._remember_events(events)

    def on_video_frame(self, payload: Any) -> None:
        frame, match_time_seconds = self._unpack_video_frame(payload)
        if frame is None or self._latest_stats is None or self._latest_events is None:
            return

        if self._dashboard_size is not None:
            width, height = self._dashboard_size
        else:
            width, height = frame.shape[1], frame.shape[0]

        dashboard = self.render(
            self._latest_stats,
            self._latest_events,
            match_time_seconds,
            width,
            height,
        )
        if self._event_bus is not None:
            self._event_bus.publish(self.output_event_type, dashboard)

    def render(
        self,
        stats: Statistics,
        events: "FrameEvents",
        match_time_seconds: float,
        width: int,
        height: int,
    ) -> "np.ndarray":
        self._remember_events(events)
        counts = dict(getattr(stats, "event_counts", {}) or self._event_counts)
        return render(
            stats,
            events,
            match_time_seconds,
            width,
            height,
            event_counts=counts,
            event_history=list(self._event_history),
        )

    def render_final_report(
        self,
        stats: Statistics,
        events: "FrameEvents",
        match_time_seconds: float,
        output_path: str,
    ) -> None:
        render_final_report(stats, events, match_time_seconds, output_path)

    def reset(self) -> None:
        self._latest_stats = None
        self._latest_events = None
        self._event_counts.clear()
        self._event_history.clear()
        self._seen_event_keys.clear()

    def _unpack_video_frame(self, payload: Any) -> Tuple[Optional["np.ndarray"], float]:
        if isinstance(payload, dict):
            frame = payload.get("frame")
            match_time_seconds = payload.get("match_time_seconds", self.match_time_seconds)
            self.match_time_seconds = float(match_time_seconds)
            return frame, self.match_time_seconds
        return payload, self.match_time_seconds

    def _remember_events(self, events: "FrameEvents") -> None:
        for event in getattr(events, "eventos", []):
            key = event_key(event)
            if key in self._seen_event_keys:
                continue
            self._seen_event_keys.add(key)
            self._event_counts[getattr(event, "type", event.__class__.__name__)] += 1
            self._event_history.append(event)


def render(
    stats: Statistics,
    events: "FrameEvents",
    match_time_seconds: float,
    width: int,
    height: int,
    *,
    event_counts: Optional[Mapping[str, int]] = None,
    event_history: Optional[Iterable[object]] = None,
) -> "np.ndarray":
    if width <= 0 or height <= 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    layout = DashboardLayout(width, height)
    draw_dashboard_sections(
        canvas,
        layout,
        stats,
        match_time_seconds,
        event_counts or get_event_counts(stats, events),
        event_history or getattr(events, "eventos", []),
    )
    return canvas


def render_final_report(
    stats: Statistics,
    events: "FrameEvents",
    match_time_seconds: float,
    output_path: str,
) -> None:
    width, height = 1280, 720
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    layout = DashboardLayout(width, height)
    draw_dashboard_sections(
        canvas,
        layout,
        stats,
        match_time_seconds,
        count_events(events),
        getattr(events, "eventos", []),
        title="FutBotMX - Reporte final",
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def render_dashboard(
    stats: Statistics,
    events: "FrameEvents",
    match_time_seconds: float,
    width: int,
    height: int,
) -> "np.ndarray":
    return DashboardRenderer().render(stats, events, match_time_seconds, width, height)
