import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request

print(f"Supervision versión: {sv.__version__}")
Path("assets").mkdir(exist_ok=True)

urllib.request.urlretrieve(
    "https://ultralytics.com/images/bus.jpg",
    "assets/bus.jpg"
)

image = cv2.imread("assets/bus.jpg")
print(f"Imagen cargada: {image.shape}")

plt.figure(figsize=(12, 7))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Imagen de prueba")
# plt.show()

# n=nano, s=small, m=medium, l=large, x=extra-large
model = YOLO("yolov8n.pt")
image = cv2.imread("assets/bus.jpg")

results = model(image)[0]

detections = sv.Detections.from_ultralytics(results)

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

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [
    f"{results.names[class_id]} {conf:.0%}"
    for class_id, conf in zip(detections.class_id, detections.confidence)
]

annotated = box_annotator.annotate(scene=image.copy(), detections=detections)
annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
plt.figure(figsize=(12, 7))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Pipeline completo: detección + anotación con Supervision")
# plt.show()

results_estricto = model(image, conf=0.1)[0]
detections_estricto = sv.Detections.from_ultralytics(results_estricto)

print(f"Con conf=0.5 (por defecto): {len(detections)} objetos detectados")
print(f"Con conf=0.8 (estricto):    {len(detections_estricto)} objetos detectados")

primera = detections[0]
x1, y1, x2, y2 = primera.xyxy[0]
print(f"Primera detección:")
print(f"  Clase:    {results.names[primera.class_id[0]]}")
print(f"  Confianza: {primera.confidence[0]:.1%}")
print(f"  Posición:  esquina superior-izquierda ({x1:.0f}, {y1:.0f})")
print(f"             esquina inferior-derecha   ({x2:.0f}, {y2:.0f})")
print(f"  Tamaño:    {x2-x1:.0f} px de ancho × {y2-y1:.0f} px de alto")

model_s = YOLO("yolov8s.pt")  # ~22 MB — más preciso, más lento
results_s = model_s(image)[0]
detections_s = sv.Detections.from_ultralytics(results_s)

print(f"yolov8n (nano):  {len(detections)} objetos")
print(f"yolov8s (small): {len(detections_s)} objetos")

urllib.request.urlretrieve(
    "https://ultralytics.com/images/zidane.jpg",
    "assets/zidane.jpg"
)

cv2.imread("assets/zidane.jpg")

import json

def detections_to_dict(detections, class_names=None):
    """Convierte sv.Detections a un dict JSON-compatible."""
    return {
        "xyxy":        detections.xyxy.tolist(),
        "confidence":  detections.confidence.tolist() if detections.confidence is not None else None,
        "class_id":    detections.class_id.tolist()   if detections.class_id   is not None else None,
        "class_names": [class_names[c] for c in detections.class_id]
                       if (class_names and detections.class_id is not None) else None,
    }

resultado = detections_to_dict(detections, class_names=results.names)

with open("assets/predicciones.json", "w", encoding="utf-8") as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)

print("Guardado: assets/predicciones.json")
print(json.dumps(resultado, indent=2))


