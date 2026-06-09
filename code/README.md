futbotmx/
├── code
│   ├── analysis                  # Análisis de datos y eventos
│   │   ├── events.py             # Detección de colisión, posesión, tiro, pase
│   │   └── stats.py              # Posesión total, distancia recorrida
│   ├── configs.py                # Configuraciones generales del proyecto
│   ├── data                      # Datos de entrada y salida
│   │   ├── outputs/              # Resultados de los scripts de vision y analisis
│   │   └── videos/               # Videos de entrada
│   ├── main.py                   # Script principal para correr el proyecto
│   ├── README.md                 # Estructura del proyecto
│   ├── vision
│   │   ├── homography.py         # Cálculo de la homografía
│   │   ├── pipeline.py           # Loop principal
│   │   └── segmentation.py       # Detecciones por frame
│   └── visualization
│       ├── dashboard.py          # Matplotlib plots
│       ├── tactical_map.py       # Draw_tactical_map, trails, heatmap
│       └── video_render.py       # Video lado a lado + canvas cenital
├── contract                      # Api de transferencia de datos
│   └── file.json 
├── docs                          # Información de la convocatoria
│   ├── Convocatoria_CopaFutBotMX-Meta-VF-20260429T020141.pdf
│   └── doc.md
├── journal.md                    # Diario de desarrollo del proyecto
├── LICENSE                       # Licencia
├── notebooks                     # Carpeta de notebooks personales
│   ├── dario
│   │   └── notebooks...          # Notebooks de dario
│   └── rojas
│       └── notebooks...          # Notebooks de rojas
└── README.md                     # Documentacion del proyecto
