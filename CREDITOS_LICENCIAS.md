# Creditos y licencias

Este archivo resume los creditos, dependencias y consideraciones de licencia del
proyecto FutBotMX Vision. Debe revisarse antes de publicar el repositorio, el
video demo o cualquier entregable final.

## Proyecto

- **Nombre:** FutBotMX Vision
- **Proposito:** sistema de vision por computadora para analizar partidos de
  futbol robotico de la Copa FutBotMX.
- **Licencia del codigo del proyecto:** MIT, segun el archivo `LICENSE` incluido
  en la raiz del repositorio.

## Autores y contribucion

- Desarrollo del pipeline de vision, tracking, analisis y visualizacion:
  equipo del proyecto FutBotMX Vision.
- Videos de prueba y material de competencia:
  Copa FutBotMX / Federacion Mexicana de Robotica, sujeto a los permisos de uso
  otorgados por la convocatoria o por los organizadores.
- Apoyo de IA generativa:
  se utilizo asistencia de IA para apoyo en programacion, documentacion,
  depuracion y organizacion del proyecto. La responsabilidad tecnica y de
  validacion final corresponde al equipo del proyecto.

## Dependencias principales

Las dependencias declaradas en `environment.yml` son:

| Dependencia | Uso en el proyecto | Licencia conocida o esperada |
| --- | --- | --- |
| Python 3.10 | Lenguaje base del proyecto | Python Software Foundation License |
| NumPy | Operaciones numericas y arreglos | BSD 3-Clause |
| OpenCV | Procesamiento de imagen, HSV, mascaras y video | Apache License 2.0 |
| PyTorch | Inferencia con GPU y soporte de modelos | BSD-style |
| TorchVision | Utilidades del ecosistema PyTorch | BSD-style |
| Supervision | Detecciones, anotacion y utilidades de vision | MIT |
| Ultralytics | Carga/ejecucion del modelo SAM y flujo de segmentacion | AGPL-3.0, salvo licencia comercial aplicable |
| ByteTrack / bytetracker | Asociacion temporal de detecciones | Verificar licencia exacta del paquete instalado |

Nota: las licencias pueden variar por version o distribucion. Para una entrega
formal, verificar la licencia exacta instalada con el gestor de paquetes usado en
el entorno final.

## Modelos y pesos

### SAM 3 / `sam3.pt`

El proyecto utiliza un archivo de pesos esperado en:

```text
code/sam3.pt
```

Este archivo no debe tratarse como codigo propio del proyecto. Debe usarse y
redistribuirse unicamente bajo los terminos oficiales publicados por Meta o por
la fuente desde la cual se obtuvo el modelo.

Antes de entregar o publicar el repositorio:

- confirmar si `sam3.pt` puede redistribuirse;
- si no puede redistribuirse, excluirlo del repositorio publico;
- documentar el enlace oficial de descarga y las instrucciones para obtenerlo;
- conservar cualquier aviso de copyright o licencia requerido por Meta.

## Material audiovisual y datos

Los videos de partidos, imagenes de cancha y capturas usadas para pruebas son
material de entrada del proyecto. Su uso depende de los permisos dados por la
convocatoria, los organizadores o los autores originales.

Antes de publicar:

- confirmar que los videos pueden mostrarse en el demo y en redes sociales;
- confirmar que los participantes, equipos o instituciones autorizan su uso si
  aplica;
- evitar publicar material sensible o no autorizado;
- mantener atribucion a Copa FutBotMX / Federacion Mexicana de Robotica cuando
  corresponda.

## Assets de notebooks

El repositorio contiene notebooks y assets de experimentacion, por ejemplo
imagenes de demostracion como `bus.jpg` o `zidane.jpg` dentro de carpetas de
notebooks.

Estos archivos parecen ser material de prueba para ejemplos de vision por
computadora y no forman parte del pipeline principal de entrega. Si el repositorio
se publica, se recomienda:

- eliminar assets externos que no sean necesarios;
- conservar solo material con licencia clara;
- documentar la fuente original de cada asset conservado.

## Obligaciones practicas para publicar

Antes de publicar una version final del proyecto:

1. Revisar que `LICENSE` tenga el nombre correcto del titular del copyright.
2. Confirmar la licencia exacta de `sam3.pt`.
3. Confirmar la licencia exacta del paquete `bytetracker` instalado.
4. Confirmar si el uso de Ultralytics bajo AGPL-3.0 es compatible con la forma de
   distribucion del proyecto, o usar una licencia comercial si aplica.
5. No subir pesos, videos o assets si sus licencias no permiten redistribucion.
6. Mantener este archivo junto con el README y el archivo `LICENSE`.

## Aviso

Este documento es una guia de atribucion y cumplimiento para el proyecto. No es
asesoria legal. Para distribucion publica, comercial o institucional, revisar las
licencias oficiales de cada dependencia y modelo.
