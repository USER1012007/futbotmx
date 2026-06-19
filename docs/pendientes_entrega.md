# Pendientes de entrega FutBotMX

Lista derivada de la convocatoria en `docs/doc.md` y del estado actual del proyecto.

1. README de entrega
   - El `README.md` actual describe principalmente la estructura.
   - Falta: enfoque técnico, arquitectura real, instalación paso a paso, comandos de reproducción, requisitos de GPU/software, capturas/GIFs, resultados, créditos, licencias de terceros y enlace al reel de Instagram.

2. Video demo final
   - Hay videos generados en `code/data/outputs/`, pero falta dejar uno como demo oficial.
   - Debe durar máximo 2 minutos.
   - Debe incluir video original + resultado segmentado/anotado + visualizaciones.
   - Debe incluir una breve explicación en texto o voz del enfoque.

3. Reel de Instagram
   - Requisito obligatorio: reel público de mínimo 30 segundos.
   - Falta generar/publicar el reel y agregar el enlace al README.

4. Validación robusta del pipeline
   - El pipeline existe, pero el tracking del balón todavía está en ajuste.
   - Falta probar en videos largos y documentar estabilidad con oclusiones, manos, cámara rotada y falsos positivos.

5. Identificación real de aliados/rivales
   - La asignación ya está limitada a `A1`, `A2`, `R1`, `R2`.
   - Falta validar en videos largos que la reidentificación por apariencia conserve correctamente los equipos cuando ByteTrack cambia IDs.

6. Eventos deportivos incompletos XXXXXX
   - Ya hay posesión, pases, goles, colisiones, fuera de cancha, reposiciones y robot detenido.
   - Faltan o están débiles: tiros a gol, intercepciones, detección de remate/intento de gol y reglas arbitrales más completas.

7. RefereeEngine
   - Existe, pero está prácticamente como stub.
   - Falta implementar penalizaciones reales o quitarlo/documentarlo como trabajo futuro.

8. Reproducibilidad
   - Hay `environment.yml`, pero faltan versiones fijas y comandos claros.
   - Falta documentar cómo obtener o ubicar `sam3.pt`.
   - El código importa `trackers`, pero el entorno declara `bytetracker`; hay que verificar que otro usuario pueda instalarlo sin ajustes manuales.

9. Scripts de ejecución limpios
   - `main.py` está hardcodeado a `video1.mp4`.
   - Falta CLI o configuración para elegir video, salida, límite de frames, device `cuda/cpu`, etc.
   - La visualización final se genera desde `test/mock_tracking_visualization.py`; conviene moverla a un script formal fuera de `test/`.

10. Métricas cuantitativas XXXXXX
    - Para categoría profesional se evalúa rendimiento.
    - Falta reportar FPS, tiempo por frame, tasa de detección del balón, pérdidas de tracking, falsos positivos, eventos detectados, etc.

11. Pruebas
    - Algunas pruebas están desactualizadas con la API actual.
    - Falta una suite mínima confiable: IO, eventos, homografía, tracking JSON, render smoke test y filtros del balón.

12. Licencias y créditos
    - Existe `LICENSE`, pero el copyright dice `/usr/local/bin`.
    - Falta corregir titular.
    - Falta atribuir SAM3, Ultralytics, Supervision, OpenCV, PyTorch, ByteTrack/tracker y videos/datos usados.

13. Higiene del repositorio
    - `sam3.pt` pesa alrededor de 3.4 GB.
    - Hay outputs, tracking y runs generados.
    - Falta decidir qué va al repo público, qué va por Git LFS y qué se genera localmente.

14. Documentación de resultados
    - Falta incluir capturas del dashboard, mapa táctico, overlay y ejemplos de eventos.
    - Falta explicar limitaciones: oclusiones por manos, cámara, pérdida de pelota y falsas detecciones.

15. Contrato de datos
    - Existe `contract/file.json` como ejemplo.
    - Falta documentar formalmente el schema del `tracking.jsonl` y eventos/estadísticas para que sea entendible por evaluadores.

## Prioridad sugerida

1. Validar tracking en videos largos.
2. Limpiar scripts de ejecución y reproducción.
3. Generar video demo oficial.
4. Completar README.
5. Publicar reel y enlazarlo.
6. Corregir licencia/créditos.
7. Preparar pruebas y métricas mínimas.
