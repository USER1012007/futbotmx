# Diarios sobre avances y descubrimientos en el desarrollo del proyecto

## 08-Jun-2026

El día de hoy nos dimos una vuelta por el sitio oficial de la Copa [FutBotMX 2026](https://www.secihti.mx/futbotmx/#copa-futbox-2026-reglas) para ver las reglas de los partidos y así tener un panorama amplio sobre qué cosas eran posibles de ser mostradas en una visualización e interpretación del output de nuestro pipeline, esto con el objetivo de ver qué exactamente era posible obtener de un partido para definir qué iba a abarcar nuestro pipeline.

Nos pareció relevante observar que la mayoría de vídeos eran muy cortos y con mucha interferencia humana en el partido por lo que, además de identificar los eventos detectables también discutimos acerca de un *modo de pánico* (similar al de un parser para evitar detectar errores en cáscada y saturar de logs al usuario) con el fin de evitar que si hay mucha interferencia de los árbitros en un partido esto afecte a qué interprete la visualización de datos y no muestre datos incorrectos.
> Darío G. (abclarry)

## 09-Jun-2026

El dia de hoy estuvimos discutiendo acerca de la estructura y el pipeline que llevaremos a cabo para el proyecto, esto con el objetivo de tener una idea clara de qué cosas vamos a hacer y cómo las vamos a hacer, además de definir qué cosas son necesarias para llevar a cabo el proyecto y qué cosas no lo son.

> Emilio R. (USER1012007)

## 11-Jun-2026

El dia de hoy desarrollamos y definimos la parte de las clases de los objetos que vamos a utilizar para el proyecto, esto con el fin de tener definida la estructura de datos que vamos a utilizar para representar los eventos y las acciones que ocurren en un partido.

> Emilio R. (USER1012007)

## 12-Jun-2026

El dia de hoy empezamos a desarrollar el sistema de subscribe/publish para el proyecto, tambien a hacer uso del script de configuraciones que contiene variables globales, tambien el desarrollo de los script que son encargados de la parte de la entrada de datos, asimismo el script que estara a cargo de escribir y leer los datos de los frames en el .jsonl. Tambien se comenzo la parte de la vision por computadora para detectar a las entidades en el campo; sin embargo, esta parte se encuentra aun en desarrollo y analisis debido a que los actuales resultados no son los esperados y estamos en busca de mejores algoritmos y/o tecnicas que nos puedan ayudara mejorar la deteccion de las entidades en el campo, por el momento estamos detectando a las entidades mediante el algoritmo que usa los canales hsv para detectar los colores de los uniformes, sin embargo, esto no es suficiente para detectar a las entidades de manera precisa debido a que hay muchos factores que pueden afectar la deteccion como la iluminacion, el angulo de la camara, entre otros factores.

> Emilio R. (USER1012007)
