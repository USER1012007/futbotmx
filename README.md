# futbotmx
SAM 3 Vi project

# Estructura del proyecto
```
futbotmx/
├── code
│   ├── domain/                   # Declaracion de entidades y objetos
│   │   ├── entities.py           # Ball, Robot, Field, Team (dataclasses)
│   │   ├── events.py             # GameEvent, Goal, Pass, Shot, Collision
│   │   └── stats.py              # MatchStats, posesión, distancia
│   │
│   ├── io
│   │   ├── video_source.py       # Abstrae .mp4 / cámara
│   │   └── tracking_io.py        # Funciones para leer y escribir archivos de tracking
│   │
│   ├── vision
│   │   ├── homography.py         # Cálculo de la homografía
│   │   ├── pipeline.py           # Loop principal
│   │   └── segmentation.py       # Detecciones por frame con SAM
│   │
│   ├── analysis                  # Escucha events, computa stats
│   │   ├── event_detector.py     # FrameResult -> dispara GameEvents
│   │   └── stats_engine.py       # Acumula MatchStats desde eventos
│   │
│   ├── infra                     # EventBus y configuración
│   │   ├── event_bus.py          # publish/subscribe
│   │   └── configs.py            # Configuraciones generales del proyecto
│   │
│   ├── data                      # Datos de entrada y salida
│   │   ├── outputs/              # Resultados de los scripts de vision y analisis
│   │   ├── tracking/             # Resultados de tracking por frame
│   │   │   └── tracking.jsonl    # Resultados de tracking en formato jsonl
│   │   └── videos/               # Videos de entrada
│   │
│   ├── main.py                   # Script principal para correr el proyecto
│   │
│   └── visualization
│       ├── dashboard.py          # Matplotlib plots
│       ├── tactical_map.py       # Draw_tactical_map, trails, heatmap
│       └── video_render.py       # Video lado a lado + canvas cenital
│   
├── contract                      # Api de transferencia de datos
│   └── file.json 
│   
├── docs                          # Información de la convocatoria
│   ├── Convocatoria_CopaFutBotMX-Meta-VF-20260429T020141.pdf
│   └── doc.md
│   
├── journal.md                    # Diario de desarrollo del proyecto
│   
├── LICENSE                       # Licencia
│   
├── notebooks                     # Carpeta de notebooks personales
│   ├── dario
│   │   └── notebooks...          # Notebooks de dario
│   └── rojas
│       └── notebooks...          # Notebooks de rojas
│   
└── README.md                     # Documentacion del proyecto
```

# Rol de cada Script

## `infra/`

Contiene las herramientas que todo el proyecto necesita para existir.

### `infra/config.py`
El único lugar donde viven los parámetros del sistema. Rutas de video, resolución de cámara, umbrales de detección, hiperparámetros de SAM3. Ningún otro módulo hardcodea constantes — todos las leen de este archivo.

### `infra/event_bus.py`
Implementa el patrón *publish/subscribe*: cualquier módulo puede **publicar** un evento (`GoalScored`, `Collision`) sin saber quién lo va a recibir, y cualquier módulo puede **suscribirse** a tipos de eventos sin saber quién los genera. Esto desacopla completamente la detección de la reacción — `vision/` y `analysis/` nunca se importan entre sí; se comunican a través del bus.

---

## `io/`

Toda interacción con el sistema de archivos y con fuentes de video está contenida aquí. El resto del proyecto no sabe si el video viene de un `.mp4`, de una cámara en vivo o de un stream remoto — eso es problema exclusivo de esta capa.

### `io/video_source.py`
Abstrae el origen del video. Expone una interfaz uniforme para leer frames, independientemente de si la fuente es un archivo grabado o una cámara en tiempo real. El resto del pipeline consume frames; este script decide de dónde vienen.

### `io/tracking_io.py`
Lee y escribe el archivo `tracking.jsonl` — el registro persistente de todas las detecciones por frame. Serializa y deserializa los resultados del pipeline de visión para que puedan consultarse, reproducirse o analizarse sin necesidad de volver a correr el modelo.

---

## `domain/`

Define qué cosas existen en el mundo del fútbol robótico y qué eventos ocurren en él. No tiene lógica algorítmica — solo declaraciones.

### `domain/entities.py`
Declara los objetos que existen en el partido: `Ball`, `Robot`, `Field`, `Team`. Son `dataclasses` puras — estructuras de datos tipadas que describen el estado del mundo en un instante dado. Ningún módulo inventa su propio formato para representar una pelota; todos usan `Ball`. Esto garantiza coherencia en todo el proyecto.

### `domain/events.py`
Declara las cosas que ocurren durante el partido: `GoalScored`, `BallPossession`, `ShotDetected`, `Collision`, `PassDetected`. Los eventos son inmutables — representan hechos que ya sucedieron. Son la moneda de cambio del `EventBus`: `analysis/` los emite, `stats/` y `visualization/` los consumen.

### `domain/stats.py`
Declara la estructura de las estadísticas acumuladas del partido: `MatchStats`. Contiene porcentaje de posesión por equipo, distancia recorrida por robot, conteo de tiros y pases, y los datos crudos para el heatmap. No calcula nada — solo define la forma que tienen las estadísticas cuando alguien más las calcula.

---

## `vision/`

Todo lo que tiene que ver con percepción visual vive aquí. Esta capa convierte píxeles en datos estructurados. No sabe qué hacer con lo que detecta — eso es trabajo de `analysis/`.

### `vision/segmentation.py`
Interfaz con SAM3. Recibe un frame crudo y devuelve un `FrameResult`: las máscaras de segmentación clasificadas en pelota, robots por equipo y campo. Es el único módulo del proyecto que habla directamente con el modelo de visión. Si mañana SAM3 se reemplaza por otro modelo, solo este archivo cambia.

### `vision/homography.py`
Convierte coordenadas de píxeles en coordenadas métricas del campo real. Calcula y aplica la matriz de homografía a partir de los puntos de calibración de la cámara. Gracias a este módulo, el resto del sistema trabaja en metros, no en píxeles — una abstracción que hace las estadísticas de distancia y posición independientes de la resolución de cámara.

### `vision/pipeline.py`
Implementa el loop principal frame por frame: lee un frame de `video_source`, lo pasa por `segmentation`, transforma coordenadas con `homography`, y persiste el resultado con `tracking_io`. No analiza ni visualiza — solo coordina el flujo de datos de visión.

---

## `analysis/`

Recibe los datos crudos de visión y los convierte en conocimiento del partido. Esta capa sabe las reglas del juego; `vision/` solo sabe ver.

### `analysis/event_detector.py`
Observa la secuencia de `FrameResult` y determina cuándo ocurre algo significativo. Implementa los algoritmos de detección de colisión, cambio de posesión, tiro a gol y pase. Cuando detecta un evento, lo publica en el `EventBus`. Es el traductor entre "lo que el modelo vio" y "lo que ocurrió en el partido".

### `analysis/stats_engine.py`
Suscrito a los eventos del `EventBus`, acumula el estado de `MatchStats` a lo largo del partido. Cada vez que llega un `GoalScored` suma un gol; cada `BallPossession` actualiza el tiempo de posesión. Al final del partido, `MatchStats` contiene el resumen completo del encuentro, listo para ser visualizado o exportado.

---

## `visualization/`

Transforma datos y estadísticas en representaciones visuales consumibles por humanos. No calcula ni detecta — solo presenta.

### `visualization/tactical_map.py`
Dibuja el mapa táctico del partido sobre una representación cenital del campo. Renderiza los trails de movimiento de cada robot, el heatmap de posiciones de la pelota y los vectores de las jugadas detectadas. Es la vista que muestra no solo dónde están las cosas, sino por dónde han pasado.

### `visualization/video_render.py`
Genera el video de salida con el canvas compuesto: el video original lado a lado con la vista cenital procesada. Superpone las detecciones, identificadores de robots y eventos en tiempo real sobre cada frame. Es el artefacto visual principal del sistema.

### `visualization/dashboard.py`
Produce los gráficos estadísticos del partido usando matplotlib: posesión por equipo, distancia recorrida, frecuencia de eventos por minuto. Consume `MatchStats` directamente y los convierte en plots exportables.

---

## `main.py`

El punto de entrada del sistema. Su único trabajo es instanciar todos los módulos, conectarlos entre sí a través del `EventBus`, y arrancar el pipeline. No contiene lógica de negocio, no toma decisiones, no procesa datos. Si `main.py` hace algo más que ensamblar y arrancar, es una señal de que algo está en el lugar equivocado.

---

## Regla de dependencia

Las dependencias solo fluyen hacia arriba en el stack:

```
main -> visualization -> analysis -> vision -> domain -> infra
                                            -> io     -> infra
```

`domain/` no importa nada del proyecto. `infra/` no importa nada del proyecto. Ninguna capa inferior conoce la existencia de las capas superiores. Esta regla garantiza que el sistema sea modificable: cambiar el modelo de visión no toca el dominio; agregar un nuevo tipo de estadística no toca la visión.

