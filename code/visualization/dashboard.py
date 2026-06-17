"""Dashboard visual de FutBotMX.

Que hace: renderiza marcador, posesion, distancias y eventos en un panel BGR.
Flujo: recibe Statistics + FrameEvents, calcula layout, dibuja secciones y publica
el frame renderizado cuando se usa conectado al EventBus.
"""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Deque, Dict, Iterable, Mapping, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on runtime environment.
    raise ImportError("visualization.dashboard requires opencv-python and numpy.") from exc

from domain.stats import PossessionPct, Score, Statistics

if TYPE_CHECKING:
    from domain.events import FrameEvents
    from infra.event_bus import EventBus


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

EVENT_PRIORITY = [
    "gol_valido",
    "gol_invalido",
    "pase",
    "colision",
    "reposicion_balon",
    "reposicion_robot",
    "sacar_robot",
    "fuera_de_cancha",
    "robot_detenido",
    "panic",
]

EVENT_COLORS: Mapping[str, Color] = {
    "gol_valido": (80, 220, 80),
    "gol_invalido": (40, 40, 255),
    "pase": (210, 120, 255),
    "colision": (40, 40, 255),
    "posesion": (0, 220, 255),
    "fuera_de_cancha": (0, 140, 255),
    "robot_detenido": (150, 150, 150),
    "reposicion_balon": (255, 180, 60),
    "reposicion_robot": (255, 130, 40),
    "sacar_robot": (40, 40, 200),
    "panic": (255, 60, 255),
}

DASHBOARD_HISTORY_MAX_EVENTS = 8

# --- Layout proportions (relative to canvas dimensions) ---
_HEADER_H_RATIO = 0.16
_POSSESSION_H_RATIO = 0.14
_DISTANCE_H_RATIO = 0.20
_EVENT_SUMMARY_H_RATIO = 0.18
_MARGIN_RATIO = 0.025
_GAP_RATIO = 0.015


class Layout:
    """Geometria precomputada del dashboard.

    Convierte proporciones del canvas a pixeles.
    Centraliza margenes, alturas, gaps y escalas de texto.
    Evita recalcular layout dentro de cada seccion.
    """

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

        # Typography scale tuned for a narrow side dashboard.
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


class DashboardRenderer:
    """Renderer de dashboard con soporte opcional de EventBus.

    Guarda las ultimas estadisticas y eventos recibidos.
    Renderiza cuando llega un video_frame.
    Publica el panel final en el evento configurado.
    """

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

    def on_frame_events(self, events: FrameEvents) -> None:
        self._latest_events = events
        self._remember_events(events)

    def on_video_frame(self, payload: Any) -> None:
        frame, match_time_seconds = self._unpack_video_frame(payload)
        if frame is None or self._latest_stats is None or self._latest_events is None:
            return

        if self._dashboard_size is not None:
            w, h = self._dashboard_size
        else:
            w, h = frame.shape[1], frame.shape[0]

        dashboard = self.render(
            self._latest_stats,
            self._latest_events,
            match_time_seconds,
            w,
            h,
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
        self._remember_events(events)
        counts = dict(getattr(stats, "event_counts", {}) or self._event_counts)
        history = list(self._event_history)
        return render(
            stats,
            events,
            match_time_seconds,
            width,
            height,
            event_counts=counts,
            event_history=history,
        )

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
            match_time_seconds = payload.get(
                "match_time_seconds",
                self.match_time_seconds,
            )
            self.match_time_seconds = float(match_time_seconds)
            return frame, self.match_time_seconds
        return payload, self.match_time_seconds

    def _remember_events(self, events: FrameEvents) -> None:
        for event in getattr(events, "eventos", []):
            key = _event_key(event)
            if key in self._seen_event_keys:
                continue
            self._seen_event_keys.add(key)
            self._event_counts[getattr(event, "type", event.__class__.__name__)] += 1
            self._event_history.append(event)


# ---------------------------------------------------------------------------
# Public rendering functions
# ---------------------------------------------------------------------------

def render(
    stats: Statistics,
    events: FrameEvents,
    match_time_seconds: float,
    width: int,
    height: int,
    *,
    event_counts: Optional[Mapping[str, int]] = None,
    event_history: Optional[Iterable[object]] = None,
) -> "np.ndarray":
    """Render a per-frame FutBotMX HUD dashboard as a BGR image."""
    if width <= 0 or height <= 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    canvas = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    lo = Layout(width, height)

    _draw_header(canvas, lo, stats, match_time_seconds)
    _draw_possession(canvas, lo, _get_possession(stats))
    _draw_distance(canvas, lo, _get_distance(stats))
    _draw_event_summary(canvas, lo, event_counts or _get_event_counts(stats, events))
    _draw_event_story(canvas, lo, event_history or getattr(events, "eventos", []))

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
    lo = Layout(width, height)

    _draw_header(
        canvas,
        lo,
        stats,
        match_time_seconds,
        title="FutBotMX - Reporte final",
    )
    _draw_possession(canvas, lo, _get_possession(stats))
    _draw_distance(canvas, lo, _get_distance(stats))
    _draw_event_summary(canvas, lo, _count_events(events))
    _draw_event_story(canvas, lo, getattr(events, "eventos", []))

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
    return DashboardRenderer().render(
        stats,
        events,
        match_time_seconds,
        width,
        height,
    )


# ---------------------------------------------------------------------------
# Section drawers
# ---------------------------------------------------------------------------

def _draw_header(
    canvas: "np.ndarray",
    lo: Layout,
    stats: Statistics,
    match_time_seconds: float,
    *,
    title: str = "FutBotMX",
) -> None:
    x, y, w, h = lo.margin, lo.header_y, lo.content_w, lo.header_h
    _panel(canvas, x, y, w, h)

    score = _get_score(stats)
    pad = max(8, lo.margin // 2)

    title_scale = _fit_text_scale(title, max(80, int(w * 0.36)), lo.scale_sm, lo.thick_md)
    title_y = y + max(22, h // 3)
    _text(canvas, title, (x + pad, title_y), title_scale, WHITE, lo.thick_md)

    label_y = y + h - max(6, h // 8)
    _text(canvas, "ALLIES", (x + pad, label_y), lo.scale_xs, ALLIES, lo.thick_sm)

    rivals_label = "RIVALS"
    rivals_size = cv2.getTextSize(rivals_label, FONT, lo.scale_xs, lo.thick_sm)[0]
    _text(
        canvas,
        rivals_label,
        (x + w - rivals_size[0] - pad, label_y),
        lo.scale_xs,
        RIVALS,
        lo.thick_sm,
    )

    score_text = f"{score.allies}  -  {score.rivals}"
    score_scale = lo.scale_xl if h >= 80 else lo.scale_lg
    score_scale = _fit_text_scale(score_text, int(w * 0.50), score_scale, lo.thick_lg)
    score_size = cv2.getTextSize(score_text, FONT, score_scale, lo.thick_lg)[0]
    score_x = x + (w - score_size[0]) // 2
    score_y = y + (h + score_size[1]) // 2
    score_y = min(score_y, y + h - max(4, h // 10))
    _text(canvas, score_text, (score_x, score_y), score_scale, WHITE, lo.thick_lg)

    time_text = _format_time(match_time_seconds)
    time_scale = _fit_text_scale(time_text, max(60, int(w * 0.24)), lo.scale_md, lo.thick_md)
    time_size = cv2.getTextSize(time_text, FONT, time_scale, lo.thick_md)[0]
    time_x = x + w - time_size[0] - pad
    time_y = y + max(22, h // 3)
    _text(canvas, time_text, (time_x, time_y), time_scale, WHITE, lo.thick_md)


def _draw_possession(
    canvas: "np.ndarray",
    lo: Layout,
    possession: PossessionPct,
) -> None:
    x, y, w, h = lo.margin, lo.possession_y, lo.content_w, lo.possession_h
    _panel(canvas, x, y, w, h)

    pad = max(8, lo.margin // 2)
    label_y = y + max(16, h // 4)
    _text(canvas, "Posesion", (x + pad, label_y), lo.scale_sm, WHITE, lo.thick_md)

    allies_pct, rivals_pct = _normalize_pct(possession.allies, possession.rivals)

    bar_x = x + pad
    bar_w = max(1, w - 2 * pad)
    bar_y = y + h // 2 - lo.bar_h // 2
    allies_w = int(round(bar_w * allies_pct / 100.0))

    cv2.rectangle(
        canvas,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + lo.bar_h),
        PANEL_ALT,
        -1,
    )
    if allies_w > 0:
        cv2.rectangle(
            canvas,
            (bar_x, bar_y),
            (bar_x + allies_w, bar_y + lo.bar_h),
            ALLIES,
            -1,
        )
    if allies_w < bar_w:
        cv2.rectangle(
            canvas,
            (bar_x + allies_w, bar_y),
            (bar_x + bar_w, bar_y + lo.bar_h),
            RIVALS,
            -1,
        )

    pct_y = y + h - max(4, h // 8)
    _text(canvas, f"Allies {allies_pct:.0f}%", (bar_x, pct_y), lo.scale_xs, ALLIES, lo.thick_sm)

    rival_text = f"Rivals {rivals_pct:.0f}%"
    rival_size = cv2.getTextSize(rival_text, FONT, lo.scale_xs, lo.thick_sm)[0]
    _text(
        canvas,
        rival_text,
        (bar_x + bar_w - rival_size[0], pct_y),
        lo.scale_xs,
        RIVALS,
        lo.thick_sm,
    )


def _draw_distance(
    canvas: "np.ndarray",
    lo: Layout,
    distance_cm: Mapping[str, float],
) -> None:
    x, y, w, h = lo.margin, lo.distance_y, lo.content_w, lo.distance_h
    _panel(canvas, x, y, w, h)

    pad = max(8, lo.margin // 2)
    label_y = y + max(16, h // 5)
    _text(canvas, "Distancia recorrida", (x + pad, label_y), lo.scale_sm, WHITE, lo.thick_md)

    if not distance_cm:
        _text(canvas, "Sin datos", (x + pad, y + h // 2 + 6), lo.scale_sm, MUTED, 1)
        return

    items = [
        (team, float(distance_cm.get(team, 0.0)))
        for team in ("allies", "rivals")
        if team in distance_cm
    ]
    if not items:
        _text(canvas, "Sin datos de equipos", (x + pad, y + h // 2 + 6), lo.scale_sm, MUTED, 1)
        return

    max_distance = max(max(v for _, v in items), 1.0)

    header_space = max(30, h // 4)
    available_h = h - header_space - pad
    row_h = max(lo.row_h_min, available_h // max(len(items), 1))

    label_col_w = max(78, int(w * 0.25))
    value_col_w = max(62, int(w * 0.14))
    bar_x = x + pad + label_col_w
    bar_w = max(1, w - 2 * pad - label_col_w - value_col_w - 4)

    for index, (label, value) in enumerate(items):
        row_y = y + header_space + index * row_h
        if row_y + row_h > y + h - 2:
            break

        bar_cy = row_y + row_h // 2
        bar_top = bar_cy - lo.bar_h // 2
        fill_w = int(round(bar_w * min(value / max_distance, 1.0)))
        color = ALLIES if "all" in label.lower() or "azul" in label.lower() else ACCENT

        label_text = _truncate_text(label, label_col_w - 8, lo.scale_xs)
        _text(canvas, label_text, (x + pad, bar_cy + 5), lo.scale_xs, WHITE, lo.thick_sm)

        cv2.rectangle(
            canvas,
            (bar_x, bar_top),
            (bar_x + bar_w, bar_top + lo.bar_h),
            PANEL_ALT,
            -1,
        )
        if fill_w > 0:
            cv2.rectangle(
                canvas,
                (bar_x, bar_top),
                (bar_x + fill_w, bar_top + lo.bar_h),
                color,
                -1,
            )

        val_text = f"{value:.0f}cm"
        value_scale = _fit_text_scale(val_text, value_col_w - 6, lo.scale_xs, lo.thick_sm)
        _text(canvas, val_text, (bar_x + bar_w + 6, bar_cy + 5), value_scale, MUTED, lo.thick_sm)


def _draw_event_summary(
    canvas: "np.ndarray",
    lo: Layout,
    counts: Mapping[str, int],
) -> None:
    x, y, w, h = lo.margin, lo.event_summary_y, lo.content_w, lo.event_summary_h
    _panel(canvas, x, y, w, h)

    pad = max(8, lo.margin // 2)
    _text(canvas, "Eventos", (x + pad, y + max(18, h // 5)), lo.scale_sm, WHITE, lo.thick_md)

    active_keys = [key for key in EVENT_PRIORITY if counts.get(key, 0) > 0]
    if not active_keys:
        active_keys = ["gol_valido", "pase", "colision", "panic"]

    active_keys = active_keys[:6]

    cols = 3
    rows = 2
    cell_w = max(1, w // cols)
    start_y = y + max(38, h // 3)
    cell_h = max(24, (h - (start_y - y) - pad) // rows)

    for index, key in enumerate(active_keys):
        row = index // cols
        col = index % cols

        cx = x + col * cell_w + pad
        cy = start_y + row * cell_h

        label = EVENT_LABELS.get(key, key)
        value = counts.get(key, 0)

        label = _truncate_text(label, cell_w - 2 * pad, lo.scale_xs)
        _text(canvas, label, (cx, cy), lo.scale_xs, MUTED, lo.thick_sm)
        _text(canvas, str(value), (cx, cy + max(18, cell_h // 2)), lo.scale_md, WHITE, lo.thick_md)


def _draw_event_story(
    canvas: "np.ndarray",
    lo: Layout,
    events: Iterable[object],
) -> None:
    x, y, w, h = lo.margin, lo.story_y, lo.content_w, lo.story_h
    _panel(canvas, x, y, w, h)

    pad = max(8, lo.margin // 2)
    _text(
        canvas,
        "Historia del partido",
        (x + pad, y + max(18, h // 7)),
        lo.scale_sm,
        WHITE,
        lo.thick_md,
    )

    recent_events = list(events)[-6:]
    if not recent_events:
        _text(canvas, "Sin eventos recientes", (x + pad, y + h // 2), lo.scale_xs, MUTED, lo.thick_sm)
        return

    row_y = y + max(42, h // 4)
    line_h = max(22, int(h * 0.12))

    for event in recent_events:
        if row_y > y + h - pad:
            break

        text = _event_to_short_text(event)
        text = _truncate_text(text, w - 2 * pad - 18, lo.scale_xs)

        color = _event_color(event)
        cv2.circle(canvas, (x + pad + 5, row_y - 5), 4, color, -1, cv2.LINE_AA)
        _text(canvas, text, (x + pad + 18, row_y), lo.scale_xs, WHITE, lo.thick_sm)

        row_y += line_h


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_events(events: FrameEvents) -> Dict[str, int]:
    return dict(
        Counter(
            getattr(event, "type", event.__class__.__name__)
            for event in getattr(events, "eventos", [])
        )
    )


def _get_event_counts(stats: Statistics, events: FrameEvents) -> Mapping[str, int]:
    counts = getattr(stats, "event_counts", None)
    return counts or _count_events(events)


def _event_key(event: object) -> tuple:
    event_type = getattr(event, "type", event.__class__.__name__)
    frame = getattr(event, "frame", None)
    robot = getattr(event, "robot", None)
    robot_id = getattr(robot, "id", None)
    position = getattr(event, "position_cm", getattr(event, "position", None))
    return (event_type, frame, robot_id, str(position))


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

    # Works both for fractions (0.58 + 0.42) and percentages (58 + 42).
    scale = 100.0 / total
    return allies * scale, rivals * scale


def _format_time(seconds: float) -> str:
    safe = max(0, int(round(seconds)))
    minutes, secs = divmod(safe, 60)
    return f"{minutes:02d}:{secs:02d}"


def _event_to_short_text(event: object) -> str:
    event_type = getattr(event, "type", event.__class__.__name__)
    prefix = _event_time_prefix(event)

    if event_type == "gol_valido":
        team = getattr(event, "team", "?")
        return f"{prefix}Gol valido {team}"

    if event_type == "gol_invalido":
        team = getattr(event, "team", "?")
        reason = getattr(event, "reason", "")
        return f"{prefix}Gol invalido {team}: {reason}"

    if event_type == "pase":
        from_id = getattr(event, "from_robot_id", getattr(event, "from_id", "?"))
        to_id = getattr(event, "to_robot_id", getattr(event, "to", "?"))
        return f"{prefix}Pase {from_id} -> {to_id}"

    if event_type == "colision":
        robots = getattr(event, "robots", [])
        ids = "/".join(getattr(robot, "id", str(robot)) for robot in robots)
        return f"{prefix}Colision {ids}"

    if event_type == "posesion":
        robot = getattr(event, "robot", None)
        rid = getattr(robot, "id", "?")
        return f"{prefix}{rid} tomo posesion"

    if event_type == "fuera_de_cancha":
        object_type = getattr(event, "object_type", getattr(event, "object", "objeto"))
        return f"{prefix}{object_type} fuera de cancha"

    if event_type == "robot_detenido":
        robot = getattr(event, "robot", None)
        rid = getattr(robot, "id", "?")
        return f"{prefix}Robot {rid} detenido"

    if event_type == "reposicion_balon":
        reason = getattr(event, "reason", "")
        return f"{prefix}Reposicion balon: {reason}"

    if event_type == "reposicion_robot":
        robot = getattr(event, "robot", None)
        rid = getattr(robot, "id", "?")
        return f"{prefix}Reposicion robot {rid}"

    if event_type == "sacar_robot":
        robot = getattr(event, "robot", None)
        rid = getattr(robot, "id", "?")
        reason = getattr(event, "reason", "")
        return f"{prefix}Robot {rid} retirado: {reason}"

    if event_type == "panic":
        reason = getattr(event, "reason", "")
        return f"{prefix}Panic: {reason}"

    return f"{prefix}{EVENT_LABELS.get(event_type, event_type)}"


def _event_time_prefix(event: object) -> str:
    timestamp_s = getattr(event, "timestamp_s", None)
    if timestamp_s is not None:
        return f"{_format_time(float(timestamp_s))} - "

    frame = getattr(event, "frame", None)
    return f"F{frame} - " if frame is not None else ""


def _event_color(event: object) -> Color:
    event_type = getattr(event, "type", event.__class__.__name__)
    return EVENT_COLORS.get(event_type, ACCENT)


def _fit_text_scale(
    text: str,
    max_px: int,
    scale: float,
    thickness: int,
    *,
    min_scale: float = 0.35,
) -> float:
    """Return a font scale that fits *text* into *max_px* without distorting it."""
    if not text or max_px <= 0:
        return scale

    current = scale
    while current > min_scale:
        text_w, _ = cv2.getTextSize(text, FONT, current, thickness)[0]
        if text_w <= max_px:
            return current
        current *= 0.92

    return min_scale


def _truncate_text(text: str, max_px: int, scale: float) -> str:
    """Truncate *text* so its rendered width stays within *max_px*."""
    if not text:
        return text

    text_w, _ = cv2.getTextSize(text, FONT, scale, 1)[0]
    if text_w <= max_px:
        return text

    suffix = "..."
    lo_i, hi_i = 0, len(text)

    while lo_i < hi_i:
        mid = (lo_i + hi_i + 1) // 2
        probe = text[:mid] + suffix
        probe_w, _ = cv2.getTextSize(probe, FONT, scale, 1)[0]

        if probe_w <= max_px:
            lo_i = mid
        else:
            hi_i = mid - 1

    return text[:lo_i] + suffix


def _panel(canvas: "np.ndarray", x: int, y: int, width: int, height: int) -> None:
    overlay = canvas.copy()
    radius = max(6, min(14, min(width, height) // 10))
    _rounded_rect(overlay, x, y, width, height, radius, PANEL, -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    _rounded_rect(canvas, x, y, width, height, radius, (58, 68, 82), 1)


def _rounded_rect(
    canvas: "np.ndarray",
    x: int,
    y: int,
    width: int,
    height: int,
    radius: int,
    color: Color,
    thickness: int,
) -> None:
    x2 = x + width
    y2 = y + height
    radius = max(1, min(radius, width // 2, height // 2))

    if thickness < 0:
        cv2.rectangle(canvas, (x + radius, y), (x2 - radius, y2), color, thickness)
        cv2.rectangle(canvas, (x, y + radius), (x2, y2 - radius), color, thickness)
        cv2.circle(canvas, (x + radius, y + radius), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, (x2 - radius, y + radius), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, (x2 - radius, y2 - radius), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, (x + radius, y2 - radius), radius, color, thickness, cv2.LINE_AA)
        return

    cv2.line(canvas, (x + radius, y), (x2 - radius, y), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x, y + radius), (x, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.line(canvas, (x2, y + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + radius, y + radius), (radius, radius), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x2 - radius, y + radius), (radius, radius), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(canvas, (x + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness, cv2.LINE_AA)


def _text(
    canvas: "np.ndarray",
    text: str,
    origin: Tuple[int, int],
    scale: float,
    color: Color,
    thickness: int,
) -> None:
    cv2.putText(canvas, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)
