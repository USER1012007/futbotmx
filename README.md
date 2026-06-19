# FutBotMX Vision

Sistema de vision por computadora para analizar partidos de futbol robotico de la
Copa FutBotMX. El proyecto usa SAM 3 como modelo base de segmentacion y lo
complementa con tracking temporal, filtros geometricos, vision clasica, homografia,
deteccion de eventos y visualizaciones tacticas.

## Objetivo

Procesar videos de futbol robotico para:

- segmentar cancha, robots y balon;
- rastrear las trayectorias de robots y pelota;
- asignar identidades estables `A1`, `A2`, `R1`, `R2`;
- calcular posesion, distancia recorrida, velocidades y eventos del partido;
- generar una visualizacion narrativa con video original, mapa tactico, marcador,
  posesion, eventos e historia del juego.

## Entregables

- Codigo funcional del pipeline de vision, tracking, analisis y visualizacion.
- Archivo de tracking por frame en `code/data/tracking/tracking.jsonl`.
- Video renderizado con dashboard en `code/data/outputs/`.
- Documentacion de arquitectura, instalacion, ejecucion, resultados y creditos en
  este README.
- Video demo oficial: `TODO: agregar ruta o enlace`.
- Reel de Instagram: `TODO: agregar enlace publico`.

## Enfoque tecnico

El sistema no hace fine-tuning de SAM 3. Se usa el modelo preentrenado `sam3.pt`
con prompts de texto y se agregan capas de post-procesamiento:

1. **SAM 3 + prompts**: segmentacion de robots, balon naranja y cancha.
2. **ByteTrack**: asociacion temporal de detecciones y persistencia de IDs crudos.
3. **Filtros de pelota**: HSV naranja, circularidad, area, contexto de cancha,
   rechazo por cercania a robots y compuerta temporal.
4. **Recuperacion por ROI**: si se pierde pelota o robot, se busca cerca de la
   ultima posicion conocida.
5. **Fallback clasico de robots**: si SAM pierde robots visibles, se buscan objetos
   no verdes sobre la mascara de cancha.
6. **Homografia**: conversion de pixeles a centimetros de cancha.
7. **Asignacion de equipos**: inicializacion por lado de cancha y bloqueo de slots
   `A1/A2/R1/R2` por tracker/cercania.
8. **Analisis deportivo**: posesion, pases, goles, colisiones, fuera de cancha,
   robots detenidos y estadisticas acumuladas.
9. **Visualizacion**: composicion de video original, mapa tactico y dashboard.

La innovacion principal no esta en entrenar un nuevo modelo, sino en integrar SAM 3
con seguimiento temporal, validaciones fisicas, recuperacion local y analisis del
partido.

## Arquitectura

```text
futbotmx/
├── code/
│   ├── main.py
│   ├── run.sh
│   ├── sam3.pt
│   ├── domain/
│   │   ├── entities.py
│   │   ├── events.py
│   │   ├── field.py
│   │   └── stats.py
│   ├── infra/
│   │   ├── configs.py
│   │   └── event_bus.py
│   ├── io_utils/
│   │   ├── video_source.py
│   │   └── tracking_io.py
│   ├── vision/
│   │   ├── segmentation.py
│   │   ├── segmentation_config.py
│   │   ├── ball_utils.py
│   │   ├── robot_utils.py
│   │   ├── roi_recovery.py
│   │   ├── detection_utils.py
│   │   ├── homography.py
│   │   ├── team_assignment.py
│   │   ├── goal_detector.py
│   │   ├── smoother.py
│   │   └── pipeline.py
│   ├── analysis/
│   │   ├── pipeline.py
│   │   ├── enricher.py
│   │   ├── event_detector.py
│   │   ├── stats_engine.py
│   │   ├── event_deduper.py
│   │   └── referee_engine.py
│   ├── visualization/
│   │   ├── video_render.py
│   │   ├── tactical_map.py
│   │   ├── dashboard.py
│   │   └── layout.py
│   ├── data/
│   │   ├── videos/
│   │   ├── tracking/
│   │   └── outputs/
│   └── test/
│       └── mock_tracking_visualization.py
├── docs/
│   ├── doc.md
│   └── Convocatoria_CopaFutBotMX-Meta-VF-20260429T020141.pdf
├── contract/
├── environment.yml
├── LICENSE
└── README.md
```

## Flujo de procesamiento

```text
main.py
  ├─ vision.pipeline.Pipeline
  │   ├─ VideoSource lee frames
  │   ├─ SegmentationEngine detecta cancha, robots y balon
  │   ├─ HomographyEngine proyecta pixeles a centimetros
  │   ├─ TeamAssigner asigna A1/A2/R1/R2
  │   ├─ PositionSmoother reduce jitter
  │   └─ TrackingIO escribe tracking.jsonl
  │
  └─ analysis.pipeline.AnalysisPipeline
      ├─ DataEnricher estima velocidades
      ├─ EventDetector detecta eventos
      ├─ StatsEngine acumula estadisticas
      └─ RefereeEngine reserva reglas arbitrales futuras
```

## Modulos principales

### `vision/`

Convierte pixeles en entidades estructuradas.

- `segmentation.py`: motor principal de SAM 3, ByteTrack, filtros y recuperacion.
- `segmentation_config.py`: constantes de prompts, clases, umbrales y ROI.
- `ball_utils.py`: filtros de pelota naranja, HSV, circularidad, campo y template
  matching.
- `robot_utils.py`: compuertas temporales de robots y fallback clasico.
- `roi_recovery.py`: busqueda local de pelota y robots perdidos.
- `homography.py`: calcula y aplica homografia hacia coordenadas metricas.
- `team_assignment.py`: asigna y estabiliza `A1/A2/R1/R2`.
- `goal_detector.py`: detecta porterias amarilla y azul por HSV.
- `pipeline.py`: coordina el flujo de vision frame por frame.

### `analysis/`

Convierte tracking en informacion deportiva.

- `enricher.py`: estima velocidad y direccion.
- `event_detector.py`: detecta posesion, pases, goles, colisiones, fuera de cancha
  y robots detenidos.
- `stats_engine.py`: acumula marcador, posesion, distancia, eventos y velocidades.
- `pipeline.py`: conecta analisis al `EventBus`.

### `visualization/`

Presenta los resultados.

- `video_render.py`: dibuja robots, pelota y eventos sobre video.
- `tactical_map.py`: genera mapa cenital de cancha.
- `dashboard.py`: marcador, posesion, distancia, eventos e historia.
- `layout.py`: distribucion visual de paneles.

### `domain/`, `infra/`, `io_utils/`

- `domain/`: entidades y eventos compartidos por todo el sistema.
- `infra/`: configuracion y bus publish/subscribe.
- `io_utils/`: lectura de video y persistencia de tracking.

## Requisitos de hardware y software

Recomendado:

- Linux.
- Python 3.10.
- GPU NVIDIA con CUDA.
- Para pruebas cortas: RTX 3050 6GB con video pequeno o `imgsz` reducido.
- OpenCV.
- PyTorch con CUDA.
- Ultralytics.
- Roboflow Supervision.
- ByteTrack/tracker compatible.

El archivo de pesos esperado es:

```text
code/sam3.pt
```

El modelo debe usarse conforme a la licencia publicada por Meta para SAM 3.

## Instalacion

Desde la raiz del repositorio:

```bash
cd code
```

Crear o activar un entorno Python 3.10 con las dependencias del proyecto.

Validar CUDA:

```bash
python3.10 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Si `torch.cuda.is_available()` devuelve `False`, SAM3 no correra en GPU.

## Configuracion del video

El video por defecto esta definido en:

```text
code/main.py
```

Linea relevante:

```python
video_path = cfg.BASE_DIR / "data/videos/video1.mp4"
```

Para procesar otro video, cambiar esa ruta por el archivo deseado dentro de
`code/data/videos/`.

## Ejecucion del pipeline

Desde `code/`:

```bash
python3 main.py > salida.log 2>&1
```

Monitorear progreso:

```bash
tail -f salida.log
```

Salida principal:

```text
code/data/tracking/tracking.jsonl
```

## Generación de visualización

Desde `code/`:

```bash
python3 test/mock_tracking_visualization.py > salida_tracking.log 2>&1
```

La visualización se genera dentro de:

```text
code/data/outputs/
```

El script `code/run.sh` ejecuta tracking y render:

```bash
cd code
bash run.sh
```

## Configuración recomendada por GPU

### RTX 3050 6GB

- Probar con clips cortos de 10 a 30 segundos.
- Preferir 720p o menos.
- Si hay error de memoria CUDA, reducir `imgsz` en `vision/segmentation.py`:

```python
imgsz=512
```

Si aun falla:

```python
imgsz=384
```

## Salidas del sistema

- `code/data/tracking/tracking.jsonl`: resultado por frame.
- `code/data/outputs/.../mock_tracking_visualization.mp4`: video final.
- `code/salida.log`: log de procesamiento.
- `code/salida_tracking.log`: log de visualizacion.

## Contrato de datos de `tracking.jsonl`

El archivo `tracking.jsonl` tiene una linea JSON por frame:

```json
{
  "frame_id": 123,
  "data": {
    "frame_id": 123,
    "timestamp_s": 4.1,
    "robots": [],
    "ball": null,
    "field_mask_present": true,
    "repositions": []
  }
}
```

### Robot

```json
{
  "id": "A1",
  "team_id": "allies",
  "tracker_id": 7,
  "position_pixel": {
    "x": 523.2,
    "y": 841.5,
    "is_metric": false,
    "unit": "px"
  },
  "position_metric": {
    "x": 85.1,
    "y": 43.2,
    "is_metric": true,
    "unit": "cm"
  },
  "speed_cm_s": 12.4,
  "angle": -35.0,
  "is_penalized": false,
  "penalization_frames_left": 0
}
```

Campos importantes:

- `id`: identidad estable (`A1`, `A2`, `R1`, `R2`).
- `team_id`: `allies` o `rivals`.
- `tracker_id`: ID temporal de ByteTrack; no debe usarse como identidad final.
- `position_pixel`: centro del robot en pixeles.
- `position_metric`: posición en centimetros por homografía.
- `speed_cm_s`: velocidad estimada.
- `angle`: dirección estimada.

### Balón

```json
{
  "id": "ball",
  "tracker_id": 2,
  "position_pixel": {
    "x": 610.0,
    "y": 740.0,
    "is_metric": false,
    "unit": "px"
  },
  "position_metric": {
    "x": 121.5,
    "y": 91.0,
    "is_metric": true,
    "unit": "cm"
  },
  "speed_cm_s": 45.2,
  "direction_vector": [12.1, -4.5]
}
```

## Visualización y narrativa

El dashboard final muestra:

- video original anotado;
- mapa táctico cenital;
- marcador;
- posesión por equipo;
- distancia recorrida;
- eventos acumulados;
- historia cronológica del partido.

Eventos considerados:

- goles;
- posesión;
- pases;
- colisiones;
- fuera de cancha;
- robots detenidos;
- reposicionamientos.

## Resultados obtenidos

El sistema produce tracking y visualizaciones completas para los videos de prueba.
En las corridas realizadas se observo:

- tracking estable de robots cuando hay visibilidad parcial;
- recuperacion de robots mediante ROI y fallback clasico;
- tracking de pelota con filtros de color, cancha y compuertas temporales;
- asignacion estable `A1/A2/R1/R2`;
- cálculo de posesión, distancia recorrida y eventos;
- dashboard final con narrativa del partido.


- frame con deteccion de 4 robots y balon;
- mapa tactico con trayectorias;
- dashboard con marcador/posesion/eventos;
- comparativa video original vs resultado anotado.

## Limitaciones conocidas

- Las manos y oclusiones pueden ocultar robots o pelota.
- La pelota puede perderse cuando sale de cancha o queda cerca de objetos naranjas.
- La homografia depende de la calidad de la mascara de cancha y de la deteccion de
  porterias.
- La asignacion inicial de equipos usa lado de cancha; despues se bloquea por slot.
- El calculo de distancia usa un filtro anti-jitter, por lo que movimientos menores
  al umbral se consideran ruido.
- `RefereeEngine` existe como punto de extension, pero las reglas arbitrales
  avanzadas quedan como trabajo futuro.

## Reproducibilidad

Para reproducir el resultado desde cero:

1. Colocar `sam3.pt` en `code/`.
2. Colocar el video en `code/data/videos/`.
3. Ajustar `video_path` en `code/main.py`.
4. Ejecutar:

```bash
cd code
MPLCONFIGDIR=/tmp/matplotlib python3.10 main.py > salida.log 2>&1
```

5. Generar visualizacion:

```bash
MPLCONFIGDIR=/tmp/matplotlib python3.10 test/mock_tracking_visualization.py > salida_tracking.log 2>&1
```

6. Revisar resultados en:

```text
data/tracking/tracking.jsonl
data/outputs/
```

## Validacion recomendada

Antes de procesar un video largo:

- correr un clip corto;
- revisar que no haya errores en `salida.log`;
- confirmar que robots visibles aparezcan como `A1/A2/R1/R2`;
- confirmar que la pelota no salte a porterias u objetos naranjas;
- revisar que las distancias no crezcan cuando los robots estan quietos.

## Uso de IA generativa

Se utilizaron asistentes de IA generativa como apoyo para depuracion,
documentacion y organizacion del codigo. El equipo conserva responsabilidad sobre
el funcionamiento del pipeline y puede explicar sus componentes.

## Creditos y dependencias

Este proyecto usa o integra:

- SAM 3 / Segment Anything Model 3, Meta AI.
- Ultralytics SAM predictor.
- Roboflow Supervision.
- OpenCV.
- PyTorch.
- ByteTrack o tracker temporal compatible.
- Videos de futbol robotico de Copa FutBotMX / Federacion Mexicana de Robotica.

Cada dependencia conserva su propia licencia. El uso de SAM 3 debe cumplir los
terminos de licencia del modelo publicados por Meta.

## Licencia

El repositorio incluye un archivo `LICENSE`. Revisar y mantener una licencia abierta
compatible con las dependencias usadas y con los terminos de SAM 3.

## Video demo y reel

- Video demo oficial: `TODO: agregar ruta o enlace final`.
- Reel de Instagram: `TODO: agregar enlace publico final`.

## Trabajo futuro

- CLI formal para elegir video, salida, device e `imgsz`.
- Reglas arbitrales mas completas.
- Metricas cuantitativas automaticas de precision/recall para tracking.
- Mejor manejo de oclusiones severas por manos.
- Empaquetado de pesos grandes con Git LFS o descarga documentada.
