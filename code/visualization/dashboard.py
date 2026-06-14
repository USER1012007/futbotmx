from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on runtime environment.
    raise ImportError("visualization.dashboard requires opencv-python and numpy.") from exc

from domain.events import FrameEvents
from infra.event_bus import EventBus
from domain.stats import PossessionPct, Score, Statistics


Color = Tuple[int, int, int]

BACKGROUND: Color = (18, 22, 28)
PANEL: Color = (31, 38, 48)
PANEL_ALT: Color = (40, 48, 59)
WHITE: Color = (245, 248, 252)
MUTED: Color = (160, 170, 182)
ALLIES: Color = (216, 180, 0)
RIVALS: Color = (60, 35, 239)
ACCENT: Color = (0, 210, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX

EVENT_LABELS: Mapping[str, str] = {
    "gol_valido": "Goles",
    "pase": "Pases",
    "colision": "Colisiones",
    "posesion": "Posesion",
    "fuera_de_cancha": "Fuera",
    "robot_detenido": "Detenidos",
    "reposicion_balon": "Rep. balon",
    "reposicion_robot": "Rep. robot",
    "gol_invalido": "Gol invalido",
    "sacar_robot": "Sacados",
    "panic": "Panic",
}


class DashboardRenderer:
    """EventBus-aware wrapper for dashboard rendering."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        *,
        statistics_event_type: str = "statistics",
        frame_events_event_type: str = "frame_events",
        video_frame_event_type: str = "video_frame",
        output_event_type: str = "dashboard",
        default_match_time_seconds: float = 0.0,
    ):
        self.output_event_type = output_event_type
        self.match_time_seconds = default_match_time_seconds
        self._event_bus = event_bus
        self._latest_stats: Optional[Statistics] = None
        self._latest_events: Optional[FrameEvents] = None

        if event_bus is not None:
            event_bus.subscribe(statistics_event_type, self.on_statistics)
            event_bus.subscribe(frame_events_event_type, self.on_frame_events)
            event_bus.subscribe(video_frame_event_type, self.on_video_frame)

    def on_statistics(self, stats: Statistics) -> None:
        self._latest_stats = stats

    def on_frame_events(self, events: FrameEvents) -> None:
        self._latest_events = events

    def on_video_frame(self, payload: Any) -> None:
        frame, match_time_seconds = self._unpack_video_frame(payload)
        if frame is None or self._latest_stats is None or self._latest_events is None:
            return

        dashboard = self.render(
            self._latest_stats,
            self._latest_events,
            match_time_seconds,
            frame.shape[1],
            frame.shape[0],
        )
        if self._event_bus is not None:
            self._event_bus.publish(self.output_event_type, dashboard)

    def render(
        self,
        stats: Statistics,
        events: FrameEvents,
        match_time_seconds: float,
        width: int,
        height: int,
    ) -> "np.ndarray":
        return render(stats, events, match_time_seconds, width, height)

    def render_final_report(
        self,
        stats: Statistics,
        events: FrameEvents,
        match_time_seconds: float,
        output_path: str,
    ) -> None:
        render_final_report(stats, events, match_time_seconds, output_path)

    def _unpack_video_frame(self, payload: Any) -> Tuple[Optional["np.ndarray"], float]:
        if isinstance(payload, dict):
            frame = payload.get("frame")
            match_time_seconds = payload.get("match_time_seconds", self.match_time_seconds)
            self.match_time_seconds = float(match_time_seconds)
            return frame, self.match_time_seconds
        return payload, self.match_time_seconds


def render(
    stats: Statistics,
    events: FrameEvents,
    match_time_seconds: float,
    width: int,
    height: int,
) -> "np.ndarray":
    """Render a per-frame FutBotMX HUD dashboard as a BGR image."""
    if width <= 0 or height <= 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)

    margin = max(12, min(width, height) // 32)
    content_w = max(1, width - 2 * margin)
    section_gap = max(10, height // 48)

    y = margin
    _draw_header(canvas, stats, match_time_seconds, (margin, y), content_w)
    y += max(58, height // 7) + section_gap

    possession_h = max(54, height // 8)
    _draw_possession(canvas, _get_possession(stats), (margin, y), content_w, possession_h)
    y += possession_h + section_gap

    distance_h = max(64, height // 7)
    _draw_distance(canvas, _get_distance(stats), (margin, y), content_w, distance_h)
    y += distance_h + section_gap

    _draw_event_counts(
        canvas,
        _count_events(events),
        (margin, y),
        content_w,
        max(1, height - y - margin),
        compact=True,
    )
    return canvas


def render_final_report(
    stats: Statistics,
    events: FrameEvents,
    match_time_seconds: float,
    output_path: str,
) -> None:
    """Render and save a final FutBotMX report snapshot as PNG."""
    width, height = 1280, 720
    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    margin = 48

    _draw_header(canvas, stats, match_time_seconds, (margin, margin), width - 2 * margin, title="FutBotMX - Reporte final")
    _draw_possession(canvas, _get_possession(stats), (margin, 170), width - 2 * margin, 110)
    _draw_distance(canvas, _get_distance(stats), (margin, 310), width - 2 * margin, 130)
    _draw_event_counts(canvas, _count_events(events), (margin, 470), width - 2 * margin, 200, compact=False)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def render_dashboard(
    stats: Statistics,
    events: FrameEvents,
    match_time_seconds: float,
    width: int,
    height: int,
) -> "np.ndarray":
    """Render a dashboard using the class wrapper for standalone callers."""
    return DashboardRenderer().render(stats, events, match_time_seconds, width, height)


def _draw_header(
    canvas: "np.ndarray",
    stats: Statistics,
    match_time_seconds: float,
    origin: Tuple[int, int],
    width: int,
    *,
    title: str = "FutBotMX",
) -> None:
    x, y = origin
    h = 92 if canvas.shape[0] >= 500 else 58
    _panel(canvas, (x, y), width, h)
    score = _get_score(stats)

    title_scale = 0.85 if h > 70 else 0.55
    score_scale = 1.45 if h > 70 else 0.82
    _text(canvas, title, (x + 18, y + 30), title_scale, WHITE, 2)
    _text(canvas, "ALLIES", (x + 18, y + h - 18), 0.5, ALLIES, 1)
    _text(canvas, "RIVALS", (x + width - 105, y + h - 18), 0.5, RIVALS, 1)

    score_text = f"{score.allies}  -  {score.rivals}"
    score_size = cv2.getTextSize(score_text, FONT, score_scale, 2)[0]
    _text(canvas, score_text, (x + (width - score_size[0]) // 2, y + h // 2 + 14), score_scale, WHITE, 2)

    time_text = _format_time(match_time_seconds)
    time_size = cv2.getTextSize(time_text, FONT, 0.7, 2)[0]
    _text(canvas, time_text, (x + width - time_size[0] - 18, y + 32), 0.7, WHITE, 2)


def _draw_possession(
    canvas: "np.ndarray",
    possession: PossessionPct,
    origin: Tuple[int, int],
    width: int,
    height: int,
) -> None:
    x, y = origin
    _panel(canvas, origin, width, height)
    _text(canvas, "Posesion", (x + 16, y + 25), 0.55, WHITE, 1)

    bar_x = x + 16
    bar_y = y + height // 2
    bar_w = max(1, width - 32)
    bar_h = 16
    allies_pct, rivals_pct = _normalize_pct(possession.allies, possession.rivals)
    allies_w = int(round(bar_w * allies_pct / 100.0))

    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), PANEL_ALT, -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + allies_w, bar_y + bar_h), ALLIES, -1)
    cv2.rectangle(canvas, (bar_x + allies_w, bar_y), (bar_x + bar_w, bar_y + bar_h), RIVALS, -1)

    _text(canvas, f"Allies {allies_pct:.0f}%", (bar_x, y + height - 14), 0.48, ALLIES, 1)
    rival_text = f"Rivals {rivals_pct:.0f}%"
    size = cv2.getTextSize(rival_text, FONT, 0.48, 1)[0]
    _text(canvas, rival_text, (bar_x + bar_w - size[0], y + height - 14), 0.48, RIVALS, 1)


def _draw_distance(
    canvas: "np.ndarray",
    distance_cm: Mapping[str, float],
    origin: Tuple[int, int],
    width: int,
    height: int,
) -> None:
    x, y = origin
    _panel(canvas, origin, width, height)
    _text(canvas, "Distancia recorrida", (x + 16, y + 25), 0.55, WHITE, 1)

    if not distance_cm:
        _text(canvas, "Sin datos", (x + 16, y + height // 2 + 8), 0.55, MUTED, 1)
        return

    items = sorted(distance_cm.items(), key=lambda item: item[0])[:6]
    max_distance = max(max(distance_cm.values()), 1.0)
    row_h = max(18, (height - 40) // max(len(items), 1))

    for index, (label, value) in enumerate(items):
        row_y = y + 48 + index * row_h
        bar_x = x + 120
        bar_w = max(1, width - 220)
        fill_w = int(round(bar_w * min(value / max_distance, 1.0)))
        color = ALLIES if "all" in label.lower() or "azul" in label.lower() else ACCENT
        _text(canvas, label, (x + 16, row_y + 5), 0.42, WHITE, 1)
        cv2.rectangle(canvas, (bar_x, row_y - 9), (bar_x + bar_w, row_y + 5), PANEL_ALT, -1)
        cv2.rectangle(canvas, (bar_x, row_y - 9), (bar_x + fill_w, row_y + 5), color, -1)
        _text(canvas, f"{value:.0f} cm", (bar_x + bar_w + 12, row_y + 5), 0.42, MUTED, 1)


def _draw_event_counts(
    canvas: "np.ndarray",
    counts: Mapping[str, int],
    origin: Tuple[int, int],
    width: int,
    height: int,
    *,
    compact: bool,
) -> None:
    x, y = origin
    _panel(canvas, origin, width, height)
    _text(canvas, "Eventos", (x + 16, y + 25), 0.55, WHITE, 1)

    primary = ["gol_valido", "pase", "colision"]
    rest = [key for key in EVENT_LABELS if key not in primary and counts.get(key, 0) > 0]
    keys = primary + ([] if compact else rest)
    if compact:
        keys = keys[:3]

    col_w = max(1, width // max(len(keys), 1))
    for index, key in enumerate(keys):
        cx = x + index * col_w
        label = EVENT_LABELS.get(key, key)
        value = counts.get(key, 0)
        _text(canvas, label, (cx + 16, y + 55), 0.45, MUTED, 1)
        _text(canvas, str(value), (cx + 16, y + min(height - 16, 95)), 1.0 if not compact else 0.78, WHITE, 2)


def _count_events(events: FrameEvents) -> Dict[str, int]:
    return dict(Counter(getattr(event, "type", event.__class__.__name__) for event in events.eventos))


def _get_score(stats: Statistics) -> Score:
    return getattr(stats, "score", Score())


def _get_possession(stats: Statistics) -> PossessionPct:
    return getattr(stats, "possession_pct", PossessionPct())


def _get_distance(stats: Statistics) -> Mapping[str, float]:
    return getattr(stats, "distance_cm", {}) or {}


def _normalize_pct(allies: float, rivals: float) -> Tuple[float, float]:
    total = allies + rivals
    if total <= 0:
        return 50.0, 50.0
    if total <= 1.0:
        return allies * 100.0 / total, rivals * 100.0 / total
    return allies * 100.0 / total, rivals * 100.0 / total


def _format_time(seconds: float) -> str:
    safe_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(safe_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def _panel(canvas: "np.ndarray", origin: Tuple[int, int], width: int, height: int) -> None:
    x, y = origin
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), PANEL, -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (58, 68, 82), 1)


def _text(
    canvas: "np.ndarray",
    text: str,
    origin: Tuple[int, int],
    scale: float,
    color: Color,
    thickness: int,
) -> None:
    cv2.putText(canvas, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)
