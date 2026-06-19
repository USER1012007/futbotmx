# FutBotMX tools

Herramientas aplicadas para ejecutar flujos del proyecto sin convertir `test/`
en punto de entrada operativo.

- `render_tracking_visualization.py`: renderiza `tracking.jsonl` sobre el video
  fuente y genera el dashboard final en `data/outputs/tracking_visualization/`.
- `diagnose_ball_tracking_video.py`: genera laminas de diagnostico para revisar
  posiciones de balon ya rastreadas.
- `diagnose_hsv_ball_candidates.py`: revisa candidatos HSV de balon en frames
  seleccionados.

Uso recomendado desde `code/`:

```bash
python tools/render_tracking_visualization.py
python tools/diagnose_ball_tracking_video.py
python tools/diagnose_hsv_ball_candidates.py
```

`test/` queda reservado para pruebas y smoke tests de componentes.
