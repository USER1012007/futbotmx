from __future__ import annotations

from typing import Iterable, Mapping

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("visualization.dashboard.sections requires opencv-python.") from exc

from domain.stats import PossessionPct, Statistics
from visualization.dashboard.events import (
    EVENT_LABELS,
    EVENT_PRIORITY,
    event_color,
    event_to_short_text,
    format_time,
    get_distance,
    get_possession,
    get_score,
    normalize_pct,
)
from visualization.dashboard.style import (
    ACCENT,
    ALLIES,
    BORDER,
    MUTED,
    PANEL,
    PANEL_ALT,
    RIVALS,
    WHITE,
    DashboardLayout,
)
from visualization.common.drawing import FONT, fit_text_scale, panel, text, truncate_text


def draw_dashboard_sections(
    canvas: "object",
    layout: DashboardLayout,
    stats: Statistics,
    match_time_seconds: float,
    event_counts: Mapping[str, int],
    event_history: Iterable[object],
    *,
    title: str = "FutBotMX",
) -> None:
    draw_header(canvas, layout, stats, match_time_seconds, title=title)
    draw_possession(canvas, layout, get_possession(stats))
    draw_distance(canvas, layout, get_distance(stats))
    draw_event_summary(canvas, layout, event_counts)
    draw_event_story(canvas, layout, event_history)


def draw_header(
    canvas: "object",
    lo: DashboardLayout,
    stats: Statistics,
    match_time_seconds: float,
    *,
    title: str = "FutBotMX",
) -> None:
    x, y, w, h = lo.margin, lo.header_y, lo.content_w, lo.header_h
    _panel(canvas, x, y, w, h)

    score = get_score(stats)
    pad = max(8, lo.margin // 2)

    title_scale = fit_text_scale(title, max(80, int(w * 0.36)), lo.scale_sm, lo.thick_md)
    title_y = y + max(22, h // 3)
    text(canvas, title, (x + pad, title_y), title_scale, WHITE, lo.thick_md)

    label_y = y + h - max(6, h // 8)
    text(canvas, "ALLIES", (x + pad, label_y), lo.scale_xs, ALLIES, lo.thick_sm)

    rivals_label = "RIVALS"
    rivals_size = cv2.getTextSize(rivals_label, FONT, lo.scale_xs, lo.thick_sm)[0]
    text(
        canvas,
        rivals_label,
        (x + w - rivals_size[0] - pad, label_y),
        lo.scale_xs,
        RIVALS,
        lo.thick_sm,
    )

    score_text = f"{score.allies}  -  {score.rivals}"
    score_scale = lo.scale_xl if h >= 80 else lo.scale_lg
    score_scale = fit_text_scale(score_text, int(w * 0.50), score_scale, lo.thick_lg)
    score_size = cv2.getTextSize(score_text, FONT, score_scale, lo.thick_lg)[0]
    score_x = x + (w - score_size[0]) // 2
    score_y = y + (h + score_size[1]) // 2
    score_y = min(score_y, y + h - max(4, h // 10))
    text(canvas, score_text, (score_x, score_y), score_scale, WHITE, lo.thick_lg)

    time_text = format_time(match_time_seconds)
    time_scale = fit_text_scale(time_text, max(60, int(w * 0.24)), lo.scale_md, lo.thick_md)
    time_size = cv2.getTextSize(time_text, FONT, time_scale, lo.thick_md)[0]
    time_x = x + w - time_size[0] - pad
    time_y = y + max(22, h // 3)
    text(canvas, time_text, (time_x, time_y), time_scale, WHITE, lo.thick_md)


def draw_possession(canvas: "object", lo: DashboardLayout, possession: PossessionPct) -> None:
    x, y, w, h = lo.margin, lo.possession_y, lo.content_w, lo.possession_h
    _panel(canvas, x, y, w, h)

    pad = max(8, lo.margin // 2)
    label_y = y + max(16, h // 4)
    text(canvas, "Posesion", (x + pad, label_y), lo.scale_sm, WHITE, lo.thick_md)

    allies_pct, rivals_pct = normalize_pct(possession.allies, possession.rivals)

    bar_x = x + pad
    bar_w = max(1, w - 2 * pad)
    bar_y = y + h // 2 - lo.bar_h // 2
    allies_w = int(round(bar_w * allies_pct / 100.0))

    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + lo.bar_h), PANEL_ALT, -1)
    if allies_w > 0:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + allies_w, bar_y + lo.bar_h), ALLIES, -1)
    if allies_w < bar_w:
        cv2.rectangle(canvas, (bar_x + allies_w, bar_y), (bar_x + bar_w, bar_y + lo.bar_h), RIVALS, -1)

    pct_y = y + h - max(4, h // 8)
    text(canvas, f"Allies {allies_pct:.0f}%", (bar_x, pct_y), lo.scale_xs, ALLIES, lo.thick_sm)

    rival_text = f"Rivals {rivals_pct:.0f}%"
    rival_size = cv2.getTextSize(rival_text, FONT, lo.scale_xs, lo.thick_sm)[0]
    text(canvas, rival_text, (bar_x + bar_w - rival_size[0], pct_y), lo.scale_xs, RIVALS, lo.thick_sm)


def draw_distance(canvas: "object", lo: DashboardLayout, distance_cm: Mapping[str, float]) -> None:
    x, y, w, h = lo.margin, lo.distance_y, lo.content_w, lo.distance_h
    _panel(canvas, x, y, w, h)

    pad = max(8, lo.margin // 2)
    label_y = y + max(16, h // 5)
    text(canvas, "Distancia recorrida", (x + pad, label_y), lo.scale_sm, WHITE, lo.thick_md)

    if not distance_cm:
        text(canvas, "Sin datos", (x + pad, y + h // 2 + 6), lo.scale_sm, MUTED, 1)
        return

    items = [
        (team, float(distance_cm.get(team, 0.0)))
        for team in ("allies", "rivals")
        if team in distance_cm
    ]
    if not items:
        text(canvas, "Sin datos de equipos", (x + pad, y + h // 2 + 6), lo.scale_sm, MUTED, 1)
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

        label_text = truncate_text(label, label_col_w - 8, lo.scale_xs)
        text(canvas, label_text, (x + pad, bar_cy + 5), lo.scale_xs, WHITE, lo.thick_sm)

        cv2.rectangle(canvas, (bar_x, bar_top), (bar_x + bar_w, bar_top + lo.bar_h), PANEL_ALT, -1)
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_top), (bar_x + fill_w, bar_top + lo.bar_h), color, -1)

        val_text = f"{value:.0f}cm"
        value_scale = fit_text_scale(val_text, value_col_w - 6, lo.scale_xs, lo.thick_sm)
        text(canvas, val_text, (bar_x + bar_w + 6, bar_cy + 5), value_scale, MUTED, lo.thick_sm)


def draw_event_summary(canvas: "object", lo: DashboardLayout, counts: Mapping[str, int]) -> None:
    x, y, w, h = lo.margin, lo.event_summary_y, lo.content_w, lo.event_summary_h
    _panel(canvas, x, y, w, h)

    pad = max(8, lo.margin // 2)
    text(canvas, "Eventos", (x + pad, y + max(18, h // 5)), lo.scale_sm, WHITE, lo.thick_md)

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

        label = truncate_text(EVENT_LABELS.get(key, key), cell_w - 2 * pad, lo.scale_xs)
        value = counts.get(key, 0)

        text(canvas, label, (cx, cy), lo.scale_xs, MUTED, lo.thick_sm)
        text(canvas, str(value), (cx, cy + max(18, cell_h // 2)), lo.scale_md, WHITE, lo.thick_md)


def draw_event_story(canvas: "object", lo: DashboardLayout, events: Iterable[object]) -> None:
    x, y, w, h = lo.margin, lo.story_y, lo.content_w, lo.story_h
    _panel(canvas, x, y, w, h)

    pad = max(8, lo.margin // 2)
    text(canvas, "Historia del partido", (x + pad, y + max(18, h // 7)), lo.scale_sm, WHITE, lo.thick_md)

    recent_events = list(events)[-6:]
    if not recent_events:
        text(canvas, "Sin eventos recientes", (x + pad, y + h // 2), lo.scale_xs, MUTED, lo.thick_sm)
        return

    row_y = y + max(42, h // 4)
    line_h = max(22, int(h * 0.12))

    for event in recent_events:
        if row_y > y + h - pad:
            break

        event_text = truncate_text(event_to_short_text(event), w - 2 * pad - 18, lo.scale_xs)
        color = event_color(event, ACCENT)
        cv2.circle(canvas, (x + pad + 5, row_y - 5), 4, color, -1, cv2.LINE_AA)
        text(canvas, event_text, (x + pad + 18, row_y), lo.scale_xs, WHITE, lo.thick_sm)
        row_y += line_h


def _panel(canvas: "object", x: int, y: int, width: int, height: int) -> None:
    panel(canvas, x, y, width, height, fill_color=PANEL, border_color=BORDER)
