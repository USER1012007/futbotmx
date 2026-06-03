import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print(f"Supervision versión: {sv.__version__}")

#######################################################################

import urllib.request

Path("assets").mkdir(exist_ok=True)

urllib.request.urlretrieve(
    "https://ultralytics.com/images/bus.jpg",
    "assets/bus.jpg"
)

image = cv2.imread("assets/bus.jpg")
print(f"Imagen cargada: {image.shape}")
# image.shape → (alto, ancho, canales)
# Los canales son BGR en OpenCV (azul, verde, rojo) — no RGB como en matplotlib

plt.figure(figsize=(12, 7))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Imagen de prueba")
#plt.show()

#######################################################################

model = YOLO("yolov8m.pt")

#######################################################################

results = model(image)[0]
# model() acepta una imagen y devuelve UNA LISTA de resultados
# [0] toma el primer (y único) resultado — siempre necesario aunque proceses una sola imagen

detections = sv.Detections.from_ultralytics(results)
# from_ultralytics() realiza la traducción al formato universal sv.Detections
# Si usáramos otro framework, el código de aquí en adelante sería IDÉNTICO

#######################################################################

print(f"Número de objetos detectados: {len(detections)}")

print(f"\n--- xyxy: coordenadas del bounding box ---")
print("Formato: [x_izquierda, y_arriba, x_derecha, y_abajo]")
print(detections.xyxy)

print(f"\n--- confidence: certeza del modelo (0 = inseguro, 1 = muy seguro) ---")
print(detections.confidence)

print(f"\n--- class_id: número de la categoría detectada ---")
print(detections.class_id)

print(f"\n--- Traducción de class_id a nombre ---")

for class_id in sorted(set(detections.class_id)):
    print(f"  Clase {class_id}: {results.names[class_id]}")

#######################################################################

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{results.names[class_id]} {conf:.0%}"
    for class_id, conf in zip(detections.class_id, detections.confidence)
]

# image.copy() es IMPORTANTE: evita modificar la imagen original
# Sin .copy(), los experimentos siguientes verán la imagen ya anotada
annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

plt.figure(figsize=(12, 7))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Pipeline completo: detección + anotación con Supervision")
plt.show()

#######################################################################

results_estricto = model(image, conf=0.8)[0]
detections_estricto = sv.Detections.from_ultralytics(results_estricto)

print(f"Con conf=0.5 (por defecto): {len(detections)} objetos detectados")
print(f"Con conf=0.8 (estricto):    {len(detections_estricto)} objetos detectados")

#######################################################################

primera = detections[0]
# sv.Detections soporta indexación igual que una lista de Python
# primera es otro sv.Detections con un solo objeto

x1, y1, x2, y2 = primera.xyxy[0]
print(f"Primera detección:")
print(f"  Clase:    {results.names[primera.class_id[0]]}")
print(f"  Confianza: {primera.confidence[0]:.1%}")
print(f"  Posición:  esquina superior-izquierda ({x1:.0f}, {y1:.0f})")
print(f"             esquina inferior-derecha   ({x2:.0f}, {y2:.0f})")
print(f"  Tamaño:    {x2-x1:.0f} px de ancho × {y2-y1:.0f} px de alto")

#######################################################################

model = YOLO("yolov8x.pt")
results_s = model_s(image, conf=0.8)[0]
detections_s = sv.Detections.from_ultralytics(results_s)

print(f"yolov8m (medium):  {len(detections)} objetos")
print(f"yolov8x (xtra large): {len(detections_s)} objetos")

#######################################################################
