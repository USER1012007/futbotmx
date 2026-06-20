# Creditos y licencias

Este archivo es la fuente de verdad para creditos, dependencias y consideraciones
de licencia del proyecto FutBotMX Vision. 

## Proyecto

- **Nombre:** FutBotMX Vision
- **Proposito:** sistema de vision por computadora para analizar partidos de
  futbol robotico de la Copa FutBotMX.
- **Licencia del codigo del proyecto:** MIT.

## Autores y contribucion

- Desarrollo del pipeline de vision, tracking, analisis y visualizacion:
  equipo del proyecto FutBotMX Vision.
- Videos de prueba y material de competencia:
  Copa FutBotMX / Federacion Mexicana de Robotica, sujeto a los permisos de uso
  otorgados por la convocatoria o por los organizadores.
- Apoyo de IA generativa:
  se utilizo asistencia de IA para apoyo en programacion, documentacion,
  depuracion y organizacion del proyecto.

## Dependencias principales

Las dependencias declaradas en [environment.yml](https://github.com/USER1012007/futbotmx/blob/main/environment.yml) son:

| Dependencia | Uso en el proyecto | Licencia |
| --- | --- | --- |
| Python 3.10 | Lenguaje base del proyecto | Python Software Foundation License |
| NumPy | Operaciones numericas y arreglos | BSD 3-Clause |
| OpenCV | Procesamiento de imagen, HSV, mascaras y video | Apache License 2.0 |
| PyTorch | Inferencia con GPU y soporte de modelos | BSD-style |
| TorchVision | Utilidades del ecosistema PyTorch | BSD-style |
| Supervision | Detecciones, anotacion y utilidades de vision | MIT |
| Ultralytics | Carga/ejecucion del modelo SAM y flujo de segmentacion | AGPL-3.0 |
| `trackers` / ByteTrack | Asociacion temporal de detecciones | MIT |

## Uso especifico de dependencias en `code/`

- `code/vision/segmentation.py` usa `SAM3SemanticPredictor` de Ultralytics para
  ejecutar SAM 3 con prompts de texto.
- `code/vision/segmentation.py` importa `from trackers import ByteTrackTracker`
  para tracking temporal de robots y balon.
- `supervision` se usa para `sv.Detections`, conversiones, filtros y combinacion
  de detecciones.
- OpenCV se usa para lectura/escritura de video, HSV, contornos, mascaras,
  homografia, diagnosticos y overlays.
- PyTorch, TorchVision y CUDA son dependencias del stack de inferencia con GPU.

## Modelos y pesos

### SAM 3 / `sam3.pt`

El proyecto utiliza un archivo de pesos esperado en:

```text
code/sam3.pt
```

- Licencia: SAM License (Meta).
- Pesos: https://huggingface.co/facebook/sam3
- Tratamiento en este proyecto: No esta incluido dentro
  del código. Se instala desde el repo/HF oficial en el paso de setup.

## Tracking: ByteTrack y paquete `trackers`

- Licencia: MIT.
- Repo: https://github.com/FoundationVision/ByteTrack
- Paper: Zhang et al. (2022), "ByteTrack: Multi-Object Tracking by Associating
  Every Detection Box."
- Tratamiento en este proyecto: Se usa `trackers` como paquete Python y `ByteTrackTracker` como clase de
tracking.

## Ultralytics

- Licencia: AGPL-3.0.
- Repo: https://github.com/ultralytics/ultralytics
- Tratamiento en este proyecto: Se usa `SAM3SemanticPredictor` para prompts de texto en `SAM3`.

## Material audiovisual y datos

Los videos de partidos, imagenes de cancha y capturas usadas para pruebas son los del dataset oficial de la [convocatoria](https://drive.google.com/drive/folders/1TF7-P4rAwPmHFw_TjmNfFU3ORxqnp8CD)

## Outputs generados

Los archivos en `code/data/tracking/` y `code/data/outputs/` son resultados
generados a partir de los videos, modelos y dependencias anteriores.

## Aviso

Este documento es una guia de atribucion y cumplimiento para el proyecto. No es
asesoria legal. Para distribucion publica, comercial o institucional, revisar las
licencias oficiales de cada dependencia y modelo.
