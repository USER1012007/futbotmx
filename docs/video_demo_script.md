# Guion para video demo FutBotMX Vision

Este documento define el guion audiovisual para el video entregable de la
convocatoria FutBotMX. El video debe durar menos de 2 minutos y mostrar el
resultado del analisis aplicado a un partido de futbol robotico.

## Objetivo del video

Explicar de forma breve y visual como FutBotMX Vision usa SAM 3 como base de
segmentacion y mejora sus resultados con una capa de vision especializada:
prompts, filtros HSV/geometricos, compuertas temporales, recuperacion por ROI,
ByteTrack, homografia y visualizacion tactica.

## Requisitos de la convocatoria cubiertos

- Duracion maxima: 2 minutos.
- Mostrar video original junto al resultado segmentado/anotado o superpuesto.
- Incluir indicadores claros de segmentacion, tracking y visualizaciones.
- Incluir una explicacion breve del enfoque usado, en voz o texto.

## Duracion objetivo

Duracion recomendada: 110 a 115 segundos.

Esto deja margen frente al limite de 120 segundos y evita cortes por exportacion o
plataformas de publicacion.

## Estructura por escenas

| Tiempo | Visual principal | Voz en off | Texto en pantalla |
| --- | --- | --- | --- |
| 0-8s | Titulo sobre clip corto del partido original. | FutBotMX Vision analiza partidos de futbol robotico usando SAM 3 como base de segmentacion. | `FutBotMX Vision` / `SAM 3 + tracking + analisis tactico` |
| 8-22s | Pantalla dividida: video original a la izquierda y resultado anotado a la derecha. | A partir del video original, el sistema detecta cancha, robots y balon, mantiene trayectorias y convierte cada frame en datos estructurados. | `Cancha` / `robots` / `balon` / `trayectorias` |
| 22-38s | Diagrama simple de flujo: video, vision, homografia, analisis, dashboard. | La arquitectura procesa el video por etapas: vision, homografia, asignacion de equipos, analisis de eventos y visualizacion tactica. | `VideoSource -> Vision -> Homography -> Analysis -> Dashboard` |
| 38-58s | Close-up de detecciones de cancha, robots y balon. | La parte clave es que no hacemos fine-tuning de SAM 3. Lo usamos con prompts de texto y agregamos post-procesamiento para el dominio del juego. | `SAM 3 sin fine-tuning` / `prompts + post-procesamiento` |
| 58-78s | Enfoque en el balon: candidato correcto y falsos positivos descartados. | Para estabilizar el balon combinamos detecciones de SAM con color HSV naranja, circularidad, area, contexto de cancha y distancia respecto a robots. | `HSV naranja` / `circularidad` / `contexto de cancha` |
| 78-94s | Ejemplo de perdida y recuperacion: recuadro ROI alrededor de la ultima posicion. | Si SAM pierde un objeto por oclusion o escala, el sistema recupera robots y balon en regiones de interes cerca de la ultima posicion conocida. | `Recuperacion por ROI` / `memoria temporal` |
| 94-106s | Mapa tactico, trayectorias e identidades `A1`, `A2`, `R1`, `R2`. | ByteTrack mantiene IDs temporales, la homografia proyecta posiciones a centimetros y la asignacion de equipos estabiliza identidades. | `ByteTrack` / `homografia` / `A1 A2 R1 R2` |
| 106-115s | Dashboard final con video anotado, mapa tactico, marcador, posesion y eventos. | El resultado es un dashboard que transforma video crudo en tracking, metricas y narrativa deportiva reproducible. | `Video crudo -> tracking -> metricas -> historia del partido` |

## Guion de voz en off

Leer en tono claro, con ritmo constante. La duracion estimada es de 110 a 115
segundos.

```text
FutBotMX Vision analiza partidos de futbol robotico usando SAM 3 como modelo base
de segmentacion.

A partir del video original, el sistema detecta la cancha, los robots y el balon,
mantiene sus trayectorias y convierte cada frame en datos estructurados.

La arquitectura procesa el video por etapas: primero la capa de vision, despues
homografia para pasar de pixeles a centimetros, asignacion de equipos, analisis
de eventos y finalmente visualizacion tactica.

La parte clave es que no hacemos fine-tuning de SAM 3. Usamos prompts de texto
para detectar robots, balon naranja y cancha verde, y encima agregamos
post-procesamiento especializado para el dominio del juego.

Para estabilizar el balon, combinamos candidatos de SAM con filtros HSV de color
naranja, circularidad, tamano, contexto de cancha y distancia respecto a robots.
Tambien usamos compuertas temporales para rechazar saltos fisicamente imposibles.

Cuando SAM pierde un objeto por oclusion o escala, el sistema no busca a ciegas:
recupera robots y balon en regiones de interes alrededor de la ultima posicion
conocida.

Luego ByteTrack mantiene IDs temporales, la homografia proyecta posiciones a la
cancha real, y la asignacion de equipos estabiliza identidades como A1, A2, R1 y
R2.

Con esas senales, el modulo de analisis calcula posesion, velocidades, distancia
recorrida y eventos como pases, goles, colisiones o fuera de cancha.

El resultado final es un dashboard que combina video anotado, mapa tactico,
marcador, posesion e historia del partido. FutBotMX Vision transforma video crudo
en tracking, metricas y narrativa deportiva reproducible.
```

## Textos cortos para pantalla

Usar pocos textos y mantenerlos visibles maximo 3 a 5 segundos.

- `SAM 3 sin fine-tuning`
- `Prompts de texto`
- `Filtro HSV naranja`
- `Circularidad + area`
- `Contexto de cancha`
- `Compuerta temporal`
- `Recuperacion por ROI`
- `ByteTrack`
- `Homografia px -> cm`
- `A1 A2 R1 R2`
- `Posesion / eventos / distancia`

## Material visual requerido

1. Clip original del partido.
2. Render con detecciones o anotaciones sobre el video.
3. Vista comparativa original vs resultado.
4. Ejemplo visual del balon detectado.
5. Ejemplo visual de recuperacion por ROI.
6. Dashboard final con mapa tactico, marcador, posesion, eventos y trayectoria.

Si no existe un render final en `code/data/outputs/`, generarlo antes de editar
el video.

## Guia de edicion

- Resolucion recomendada: 1920x1080.
- Formato final: MP4.
- Duracion maxima: 120 segundos.
- Duracion recomendada: 110-115 segundos.
- Audio: voz clara, sin musica que tape la narracion.
- Texto: alto contraste, pocas palabras y sin cubrir robots o balon.
- Ritmo: cortes rapidos, pero dejar suficiente tiempo para entender cada tecnica.
- Primera mitad: explicar problema y vision.
- Segunda mitad: mostrar estabilizacion, tracking, analisis y dashboard.

## Checklist de validacion

Antes de exportar el video final, verificar:

- El archivo dura menos de 2 minutos.
- Aparece video original y resultado segmentado/anotado.
- Se ve al menos una comparacion lado a lado o superpuesta.
- Se menciona explicitamente que SAM 3 se usa sin fine-tuning.
- Se mencionan prompts, filtros HSV/geometricos, compuertas temporales, ROI,
  ByteTrack y homografia.
- Los textos son legibles en pantalla de celular.
- La voz se entiende sin depender de subtitulos.
- El cierre muestra el dashboard final.
- El archivo final queda enlazado en el README.

## Version corta para reel

Para el reel de Instagram de minimo 30 segundos, reutilizar estas ideas:

1. 0-5s: original vs resultado.
2. 5-12s: SAM 3 sin fine-tuning + prompts.
3. 12-22s: filtros de balon, ROI y tracking.
4. 22-30s: dashboard final y metricas.

Texto sugerido para reel:

```text
Usamos SAM 3 sin fine-tuning para segmentar futbol robotico.
Lo estabilizamos con filtros HSV, geometria, ByteTrack y recuperacion por ROI.
El resultado: tracking, posesion, eventos y dashboard tactico.
```
