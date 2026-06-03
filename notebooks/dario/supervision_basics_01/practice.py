import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
import json

model = YOLO("yolov8x.pt")

urllib.request.urlretrieve(
    "https://ultralytics.com/images/zidane.jpg",
    "assets/zidane.jpg"
)

mi_imagen = cv2.imread("assets/zidane.jpg")

plt.figure(figsize=(12, 7))
plt.imshow(cv2.cvtColor(mi_imagen, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Imagen de prueba")

results = model(mi_imagen)[0] 
detections = sv.Detections.from_ultralytics(results)

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
