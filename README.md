# FutBotMX Vision

Pipeline de visión por computadora para analizar partidos de fútbol robótico de
la Copa FutBotMX. Usa SAM 3 para segmentación y agrega tracking temporal,
filtros geométricos, homografía, detección de eventos y una visualización
táctica del partido.

Proyecto desarrollado para la **categoría Amateur**. El objetivo principal fue
construir un flujo funcional y documentado con SAM 3 preentrenado, aprender sus
fallas en video real de fútbol robótico y reforzarlo con postprocesamiento
simple, sin fine-tuning.

El sistema rastrea:

- robots con IDs estables: `A1`, `A2`, `R1`, `R2`;
- la pelota naranja;
- posesión por equipo, goles, colisiones, robots detenidos y distancia recorrida;
- un video final con frame original, mapa táctico y estadísticas del partido.

> Este proyecto no hace fine-tuning de SAM 3. Usa el modelo preentrenado
> `sam3.pt` y mejora la confiabilidad con postprocesamiento, recuperación local
> y validación temporal.

## Demo

Vista previa del tracking y dashboard final:

![Demo de FutBotMX Visión](code/data/videos/tracking_visualization.gif)

## Videos Generados

- [Video público de máximo 2 minutos](https://drive.google.com/file/d/16Tf50qdmgZ01EZD6rci76g1oS-FLVeVQ/view?usp=drive_link):
  muestra el video original junto con segmentación, tracking, mapa táctico,
  dashboard y una descripción breve del enfoque.
- [Reel público de Instagram](https://www.instagram.com/reel/DZyUy8qJg_0/?igsh=MTdwN2N3djdsZ3Ezaw==): reel de mínimo 30 segundos en el
  que se aprecia el resultado.

## Inicio Rápido

### 1. Clonar

```bash
git clone https://github.com/USER1012007/futbotmx.git
cd futbotmx
```

### 2. Crear Entorno

El proyecto está pensado para correr con GPU NVIDIA y CUDA. Linux es el entorno
recomendado para procesar videos largos; Windows también funciona si tienes
drivers NVIDIA/CUDA correctos. En macOS se puede instalar para pruebas, pero la
inferencia será más lenta si corre en CPU.

#### Linux

```bash
conda env create -f environment.yml
conda activate futbotmx
```

Validar GPU:

```bash
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

#### Windows

Usa Anaconda Prompt o PowerShell con Conda disponible:

```powershell
conda env create -f environment.yml
conda activate futbotmx
```

Validar GPU:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Si usas Windows nativo, ejecuta los comandos de Python manualmente. `run.sh`
requiere Git Bash, WSL o una terminal compatible con Bash.

#### macOS

`environment.yml` incluye `pytorch-cuda`, que no está disponible en macOS. Para
pruebas locales, crea un entorno sin CUDA:

```bash
conda create -n futbotmx python=3.10 numpy opencv pytorch torchvision -c pytorch -c conda-forge
conda activate futbotmx
pip install supervision ultralytics trackers
```

Ejecuta el pipeline con CPU:

```bash
python3 main.py --device cpu --video ruta/de_tu/video.mp4
```

### 3. Agregar Modelo y Video

Descarga los pesos oficiales de SAM 3 desde Hugging Face o el repositorio
oficial de Meta y colócalos aquí:

```text
code/sam3.pt
```

El archivo `sam3.pt` no se redistribuye como parte del código fuente por tamaño
y licencia. El pipeline espera encontrarlo en esa ruta; si no existe, la
inferencia no puede iniciar.

Después coloca un video de la Copa FutBotMX en `code/data/videos/` o usa una
ruta absoluta con `--video`.

### 4. Ejecutar Tracking

```bash
cd code
python3 main.py --video ruta/de_tu/video.mp4
```

Esto genera:

```text
data/tracking/tracking.jsonl
```

### 5. Renderizar Visualización

```bash
python3 tools/render_tracking_visualization.py \
  --video ruta/de_tu/video.mp4 \
  --tracking data/tracking/tracking.jsonl
```

El video de salida se escribe en:

```text
data/outputs/tracking_visualization/tracking_visualization.mp4
```

## Ejecución En Un Comando

Desde `code/`, ejecuta tracking y render en un solo paso:

```bash
bash run.sh --video ruta/de_tu/video.mp4
```

Para una prueba corta:

```bash
bash run.sh --video ruta/de_tu/video.mp4 --max-frames 300 --imgsz 640
```

### GPU / Tamaño De Imagen

Para una GPU pequeña:

```bash
python3 main.py --imgsz 512
```

Para una GPU más fuerte:

```bash
python3 main.py --imgsz 1280
```

También puedes usar variables de entorno:

```bash
FUTBOT_SAM_DEVICE=cuda FUTBOT_SAM_IMGSZ=1280 python3 main.py
```

### Logs

```bash
python3 main.py > salida.log 2>&1
tail -f salida.log
```

## Requisitos

Entorno recomendado:

- Linux
- Python 3.10
- GPU NVIDIA con CUDA
- PyTorch con CUDA
- OpenCV
- Ultralytics
- Supervision
- Tracker compatible con ByteTrack

## Estructura Del Proyecto

```text
futbotmx/
├── code/
│   ├── main.py
│   ├── run.sh
│   ├── vision/          # segmentación, tracking, homografía, equipos
│   ├── analysis/        # eventos, estadísticas, lógica arbitral
│   ├── visualization/   # dashboard, mapa táctico, render final
│   ├── domain/          # entidades compartidas y modelo de cancha
│   ├── io_utils/        # lectura de video y tracking I/O
│   ├── tools/           # scripts de render y diagnóstico
│   └── data/
│       ├── videos/
│       ├── tracking/
│       └── outputs/
├── docs/
├── environment.yml
├── LICENSE
├── CREDITOS_LICENCIAS.md
└── README.md
```

## Cómo Funciona

```text
Frame de video
  ├─ segmentación con SAM 3
  ├─ filtrado de robots y pelota
  ├─ asociación temporal con ByteTrack
  ├─ recuperación local por ROI cuando desaparecen objetos
  ├─ homografía a coordenadas de cancha
  ├─ IDs estables de equipo: A1/A2/R1/R2
  ├─ detección de eventos y estadísticas
  └─ render del dashboard
```

Componentes principales:

- `vision/segmentation.py`: inferencia con SAM 3, ByteTrack, recuperación de
  objetos y detecciones finales por frame.
- `vision/ball_utils.py`: filtros de pelota naranja, fallback HSV, rechazo de
  piel/manos, contexto de cancha y validación temporal.
- `vision/robot_utils.py`: compuertas temporales de robots y fallback para
  robots perdidos.
- `vision/team_assignment.py`: IDs estables y slots por equipo.
- `vision/homography.py`: proyección de píxeles a centímetros.
- `analysis/event_detector.py`: goles, pases, posesión, colisiones, fuera de
  cancha y robots detenidos.
- `analysis/stats_engine.py`: estadísticas del partido y métricas acumuladas.
- `visualization/`: mapa táctico y video final con dashboard.

## Experimentación Amateur

El enfoque no busca entrenar un modelo nuevo. La contribución está en integrar
SAM 3 con un pipeline reproducible y estudiar sus errores en videos reales:

- Prompts de texto para campo, robots y balón naranja.
- Asociación temporal con ByteTrack para mantener identidades entre frames.
- Filtros HSV y de forma para recuperar la pelota cuando SAM 3 la pierde.
- Búsqueda local por ROI *(Region of Interest)* alrededor de objetos recientes.
- Rechazo de falsos positivos por manos, piel u objetos naranjas fuera de la
  cancha (contexto de cancha).
- Homografía para proyectar posiciones de píxeles a centímetros.
- Dashboard con posesión, distancia, marcador de goles y eventos.
- Tactical Map con trayectorias (rastros), cruces de colisión y vector de dirección de la pelota.
- Estado de pánico cuando hay inestabilidades bruscas en el tracking (basado en el modo de pánico del
  parser de un compilador para no detectar errores en cascada).
- Detección de eventos con cooldowns por tipo para evitar duplicados en ráfagas de frames.

## Archivos de Salida

```text
code/data/tracking/tracking.jsonl
code/data/outputs/tracking_visualization/tracking_visualization.mp4
code/salida.log
code/salida_tracking.log
```

`tracking.jsonl` guarda un objeto JSON por frame:

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

Los IDs de robots se normalizan a:

```text
A1, A2, R1, R2
```

El `tracker_id` crudo viene de ByteTrack y no debe usarse como identidad final
del robot.

## Resultados Incluidos

El repositorio incluye una muestra de tracking ya generada en:

```text
code/data/tracking/tracking.jsonl
```

Esa muestra corresponde a `video1.mp4` y contiene:

- 254 frames procesados a 30 FPS.
- 505 detecciones de robots guardadas.
- balón detectado o recuperado en 180 frames.
- IDs estables `A1` y `R1` en la muestra corta.

La muestra sirve para inspeccionar el formato de salida y renderizar la
visualización sin reprocesar todo el video.

## Limitaciones Conocidas

- Las oclusiones fuertes por manos pueden ocultar la pelota o los robots.
- La pelota puede desaparecer cuando sale de la cancha o queda totalmente
  cubierta.
- La calidad de la homografía depende de la visibilidad de la cancha y porterías.
- La asignación de equipos inicia por lado de cancha y después se bloquea por
  slots estables.
- La métrica de distancia ignora intencionalmente movimientos pequeños por jitter.
- En algunas tomas la proyección métrica puede colocar objetos cerca o fuera del
  borde de la cancha cuando la máscara del campo o la orientación no son
  suficientemente estables.
- Los robots re-integrados al partido luego de haber sido sacados por los árbitros pueden
  ser detectados como del equipo contrario, sucede también cuando se salen de la toma y vuelven
  (aunque estén dentro de la cancha).
- La dirección de robots y balón se almacena en el JSON de cada frame. Tras evaluar los resultados,
  la dirección de los robots presentó demasiado ruido para ser informativa en la visualización,
  por lo que se optó por mostrar únicamente la del balón.
- La muestra `tracking.jsonl` incluida es corta y no cubre todos los casos del
  partido ni todos los robots esperados.

## Aprendizajes

- SAM 3 segmenta bien objetos grandes como cancha y robots, pero la pelota
  naranja puede perderse por tamaño, movimiento, sombras u oclusiones.
- Combinar SAM 3 con reglas sencillas de color (HSV), forma y contexto mejora mucho
  la estabilidad del balón.
- El tracking necesita validación temporal; aceptar cada detección cruda produce
  saltos falsos.
- La homografía aporta métricas útiles, pero es sensible a cámaras inclinadas,
  campo parcialmente visible y detecciones incompletas.
- Durante las Notebooks pudimos confirmar como YOLO (para generar bboxes) + SAM 3 para segmentar
  eran un dúo muy dinámico, sin embargo las clases de COCO no incluían una para los robots o la pelota.

## Entregables

- Pipeline funcional de tracking.
- `tracking.jsonl` con datos de robots y pelota por frame.
- Video final de visualización con dashboard.
- Documentación open source en este README.
- Licencia y créditos de terceros en [LICENSE](https://github.com/USER1012007/futbotmx/blob/main/LICENSE) y [THIRD_PARTY_LICENSES](https://github.com/USER1012007/futbotmx/blob/main/CREDITOS_LICENCIAS.md).
- Video demo y reel documentados en la sección [Videos Generados](#videos-generados).

## Integrantes

- Garcia Norrigan Luis Darío
- Rojas Badillo Emilio

## Créditos

Este proyecto usa o integra:

- SAM 3 / Segment Anything Model 3, Meta AI.
- Ultralytics SAM predictor.
- Roboflow Supervision.
- OpenCV.
- PyTorch.
- ByteTrack o tracker temporal compatible.
- Videos de Copa FutBotMX / Federación Mexicana de Robótica, sujetos a los
  permisos otorgados por los organizadores.

Consulta [THIRD_PARTY_LICENSES](https://github.com/USER1012007/futbotmx/blob/main/CREDITOS_LICENCIAS.md) para notas de licencia y atribución de terceros.

## Licencia

El código del proyecto se distribuye bajo la licencia incluida en [LICENSE](https://github.com/USER1012007/futbotmx/blob/main/LICENSE).
