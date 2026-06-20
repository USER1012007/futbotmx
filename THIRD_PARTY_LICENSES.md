# Créditos y licencias
Este archivo es la fuente de verdad para créditos, dependencias y consideraciones
de licencia del proyecto FutBotMX Vision.

## Proyecto
- **Nombre:** FutBotMX Vision
- **Propósito:** sistema de vision por computadora para analizar partidos de
  futbol robótico de la Copa FutBotMX.
- **Licencia del código del proyecto:** MIT.

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
| Ultralytics | Carga/ejecucion del modelo SAM 3 y flujo de segmentacion | AGPL-3.0 |
| `trackers` / ByteTrack | Asociacion temporal de detecciones | Apache 2.0 |

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
- **Licencia:** SAM License (Meta) — licencia custom de Meta; permisiva para
  investigacion y uso no comercial, pero no es MIT/Apache/GPL estandar.
- **Repo oficial (codigo):** https://github.com/facebookresearch/sam3
- **Pesos (HuggingFace, acceso restringido):** https://huggingface.co/facebook/sam3
- **Tratamiento en este proyecto:** No esta incluido dentro del codigo.
  Los pesos son gated en HuggingFace: se requiere solicitar acceso y
  descargar `sam3.pt` manualmente antes del paso de setup.

## Tracking: ByteTrack y paquete `trackers`
- **Licencia:** Apache 2.0.
- **Repo del paquete Python:** https://github.com/roboflow/trackers
- **Algoritmo de referencia:** Zhang et al. (2022), "ByteTrack: Multi-Object
  Tracking by Associating Every Detection Box." (ECCV 2022)
- **Tratamiento en este proyecto:** Se usa `trackers` como paquete Python
  (reimplementacion limpia de Roboflow) y `ByteTrackTracker` como clase de
  tracking.

## Ultralytics
- **Licencia:** AGPL-3.0. Para cumplimiento: este proyecto es open-source y
  el codigo se publica publicamente, lo cual satisface la condicion de
  copyleft de AGPL-3.0.
- **Repo:** https://github.com/ultralytics/ultralytics
- **Tratamiento en este proyecto:** Se usa `SAM3SemanticPredictor` para
  segmentacion con prompts de texto mediante SAM 3.

## Material audiovisual y datos
Los videos de partidos, imagenes de cancha y capturas usadas para pruebas son
los del dataset oficial de la convocatoria
[aquí](https://drive.google.com/drive/folders/1TF7-P4rAwPmHFw_TjmNfFU3ORxqnp8CD).

## Outputs generados
Los archivos en `code/data/tracking/` y `code/data/outputs/` son resultados
generados a partir de los videos, modelos y dependencias anteriores.
