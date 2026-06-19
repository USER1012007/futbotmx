# Pendientes de entrega FutBotMX

Lista actualizada a partir de la convocatoria en `docs/doc.md` y de la revision
del proyecto activo `futbotmx`.

## Ya observado en el proyecto

- Pipeline principal de vision, tracking, analisis y visualizacion en `code/`.
- `README.md` con enfoque tecnico, arquitectura, instalacion, ejecucion,
  limitaciones, resultados en texto, reproducibilidad y creditos generales.
- `environment.yml` con dependencias base.
- `code/data/tracking/tracking.jsonl` como ejemplo de salida.
- Documentos auxiliares en `code/docs/`: reproduccion, contrato de tracking y
  explicacion de funciones de vision.
- Archivo de atribuciones creado en `CREDITOS_LICENCIAS.md`.
- Guion audiovisual para el video demo en `docs/video_demo_script.md`.

## Pendientes criticos antes de cerrar entrega

1. Video demo oficial
   - No se observa un `.mp4` final en `code/data/outputs/`; solo existe
     `.gitkeep`.
   - Ya existe guion de produccion en `docs/video_demo_script.md`.
   - La convocatoria exige un video de maximo 2 minutos.
   - Debe mostrar video original junto al resultado segmentado/anotado o
     superpuesto.
   - Debe incluir indicadores claros de segmentacion, tracking y visualizaciones.
   - Debe incluir una explicacion breve del enfoque en texto o voz.
   - Agregar ruta o enlace final al README.

2. Reel de Instagram
   - Sigue como `TODO` en el README.
   - Debe ser publico y durar al menos 30 segundos.
   - Agregar el enlace final al README.

3. Video de entrada reproducible
   - `code/main.py` usa `code/data/videos/video1.mp4` por defecto, pero ya
     acepta `--video`.
   - En el proyecto revisado solo se observa `code/data/videos/futbot-01.jpg`.
   - Falta agregar el video requerido, documentar su descarga o cambiar la ruta
     por un archivo existente.

4. Pesos de SAM 3
   - El codigo espera `code/sam3.pt`.
   - No se observa `sam3.pt` versionado y `.gitignore` excluye `*.pt`.
   - Falta documentar el enlace oficial, permisos de acceso, ubicacion esperada y
     restricciones de redistribucion del modelo.

5. Dependencias reproducibles
   - `environment.yml` no fija versiones.
   - `environment.yml` declara `trackers`, pero falta validar una instalacion
     limpia con la version exacta requerida por `vision.segmentation`.
   - Los notebooks usan dependencias no declaradas, como `matplotlib`,
     `huggingface_hub` y `tqdm`.
   - Falta validar que una instalacion limpia pueda correr `main.py` sin ajustes
     manuales.

6. Capturas, GIFs o evidencia visual en README
   - La convocatoria pide resultados con capturas de pantalla o GIFs.
   - El README describe resultados, pero no se observan imagenes o GIFs enlazados
     para dashboard, mapa tactico, overlay o eventos.

7. Licencia del proyecto
   - Existe `LICENSE`, pero el titular aparece como `/usr/local/bin`.
   - Falta reemplazarlo por el titular real del equipo o institucion.
   - `CREDITOS_LICENCIAS.md` deja la atribucion de terceros, pero todavia falta verificar
     licencias exactas de `sam3.pt`, Ultralytics y `trackers`.

8. Permisos y procedencia de assets
   - Hay assets de notebooks (`bus.jpg`, `zidane.jpg`, `predicciones_mascaras.json`,
     `futbot-notebooks.zip`, outputs `runs/segment/`) que no parecen necesarios
     para el pipeline final.
   - Falta decidir si se conservan, se eliminan o se mueven fuera de la entrega.
   - Para cada asset conservado debe quedar clara su fuente y licencia.

9. Metricas cuantitativas
   - Para categoria profesional se evalua rendimiento e innovacion.
   - No se observa reporte de FPS, tiempo por frame, tasa de deteccion del balon,
     perdidas de tracking, falsos positivos o estabilidad de IDs.
   - Falta agregar una tabla o seccion de metricas con el video usado.

10. Validacion en videos largos
    - Falta evidencia de pruebas en partidos completos o clips largos.
    - Validar o documentar fallas con oclusiones, manos, camara inclinada,
      cambios de iluminacion y objetos naranjas falsos.

11. Identificacion de aliados/rivales
    - Ya existe asignacion `A1`, `A2`, `R1`, `R2`.
    - Falta validar en videos largos que la identidad se mantenga cuando
      ByteTrack pierde o reasigna IDs.

12. Eventos deportivos y reglas arbitrales
    - Ya hay posesion, pases, goles, colisiones, fuera de cancha, reposiciones y
      robot detenido.
    - Faltan o estan debiles tiros a gol, intercepciones, remates/intentos de gol
      y reglas arbitrales completas.
    - `RefereeEngine` sigue documentado como punto de extension; implementar o
      dejarlo explicitamente como trabajo futuro.

13. CLI o configuracion limpia
    - `main.py` ya acepta CLI para elegir video, tracking, limite de frames,
      device `cuda/cpu` e `imgsz`.
    - La visualizacion final ya debe generarse desde
      `code/tools/render_tracking_visualization.py`.
    - Falta validar ese flujo oficial en una instalacion limpia.

14. Pruebas y validaciones limpias
    - `code/tools/check_core_components.py` valida componentes centrales con
      datos sinteticos y temporales.
    - Falta ampliar validaciones limpias para homografia, render y filtros de
      balon sin depender de videos locales o pesos SAM.

15. Higiene de repositorio publico
    - `.gitignore` excluye pesos y videos, pero `tracking.jsonl`, notebooks,
      assets y salidas de experimentos pueden quedar versionados.
    - Falta decidir que entra al repositorio publico, que va por Git LFS y que se
      genera localmente.
    - `code/docs/` aparece sin seguimiento en git en esta copia; confirmar si debe
      agregarse antes de entregar.

## Prioridad sugerida

1. Agregar o documentar `video1.mp4` y `sam3.pt`.
2. Validar que el entorno limpio instala `trackers` y corre `main.py`.
3. Generar video demo oficial y publicarlo o dejarlo en la ruta final.
4. Publicar reel de Instagram y enlazarlo.
5. Agregar capturas/GIFs y metricas al README.
6. Corregir `LICENSE` y verificar licencias exactas de modelo/dependencias.
7. Limpiar assets/notebooks que no formen parte de la entrega final.
