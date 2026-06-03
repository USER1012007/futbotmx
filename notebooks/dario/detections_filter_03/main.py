import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path
from ultralytics.models.sam import SAM3SemanticPredictor

Path("assets").mkdir(exist_ok=True)
urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", "assets/bus.jpg")
image = cv2.imread("assets/bus.jpg")

model = YOLO("../yolov8x.pt")
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

def mostrar(det, titulo, img=None):
    """Visualiza detecciones — helper para no repetir código de matplotlib en cada celda."""
    if img is None:
        img = image
    etiquetas = [f"{results.names[c]}" for c in det.class_id]
    scene = box_annotator.annotate(scene=img.copy(), detections=det)
    scene = label_annotator.annotate(scene=scene, detections=det, labels=etiquetas)
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"{titulo}  ({len(det)} objetos)")
    plt.show()

# mostrar(detections, "Todas las detecciones (punto de partida)")

#######################################################################

# La comparación crea una máscara booleana (array de True/False) para cada detección
# sv.Detections[mascara_booleana] devuelve solo las filas donde la máscara es True
mascara = detections.confidence > 0.5
alta_confianza = detections[mascara]

print(f"Total: {len(detections)} | Con confianza > 0.5: {len(alta_confianza)}")
# mostrar(alta_confianza, "Solo confianza > 0.5")

#######################################################################

# Clases disponibles en esta imagen (COCO dataset)
print("Objetos detectados:")
for class_id in sorted(set(detections.class_id)):
    n = (detections.class_id == class_id).sum()
    print(f"  Clase {class_id} ({results.names[class_id]}): {n} detecciones")

# Clase 0 = 'person' en COCO
personas = detections[detections.class_id == 0]
print(f"\nSolo personas: {len(personas)}")
# mostrar(personas, "Solo personas (clase 0)")

#######################################################################

# Para combinar condiciones, DEBES usar & (AND elemento a elemento), no 'and'
# 'and' en Python es para valores booleanos individuales, no arrays
personas_seguras = detections[
    (detections.class_id == 0) & (detections.confidence > 0.8)
]
print(f"Personas con confianza > 80%: {len(personas_seguras)}")
# mostrar(personas_seguras, "Personas con confianza > 80%")

#######################################################################

# Generamos duplicados artificiales: mismo modelo, dos umbrales de confianza diferentes
# conf=0.3 detecta más objetos (incluyendo muchos con baja confianza)
# conf=0.7 detecta menos pero con mayor certeza
# Al mergear ambos, obtenemos el mismo objeto detectado dos veces
results_baja = model(image, conf=0.3)[0]
results_alta = model(image, conf=0.7)[0]
det_baja = sv.Detections.from_ultralytics(results_baja)
det_alta = sv.Detections.from_ultralytics(results_alta)

mezclado = sv.Detections.merge([det_baja, det_alta])
print(f"Detecciones individuales: baja_conf={len(det_baja)}, alta_conf={len(det_alta)}")
print(f"Después de merge (duplicados incluidos): {len(mezclado)}")

# threshold=0.5 → si dos cajas se solapan más del 50%, NMS elimina la de menor confianza
sin_duplicados = mezclado.with_nms(threshold=0.5)
print(f"Después de NMS (sin duplicados): {len(sin_duplicados)}")

# mostrar(mezclado,       "Antes de NMS (con duplicados)")
# mostrar(sin_duplicados, "Después de NMS (threshold=0.5)")

#######################################################################

# detections.area calcula automáticamente el área de cada bounding box en píxeles²
areas = detections.area
print(f"Área mínima:  {areas.min():.0f} px²")
print(f"Área máxima:  {areas.max():.0f} px²")
print(f"Área promedio: {areas.mean():.0f} px²")

# Objetos muy pequeños pueden ser falsos positivos o ruido
objetos_grandes = detections[detections.area > 5000]
print(f"\nObjetos con área > 5000 px²: {len(objetos_grandes)}")
mostrar(objetos_grandes, "Solo objetos grandes (área > 5000 px²)")

#######################################################################

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, thresh in zip(axes, [0.3, 0.5, 0.8]):
    filtered = sv.Detections.merge([det_baja, det_alta]).with_nms(threshold=thresh)
    etiquetas = [results.names[c] for c in filtered.class_id]
    scene = box_annotator.annotate(scene=image.copy(), detections=filtered)
    scene = label_annotator.annotate(scene=scene, detections=filtered, labels=etiquetas)
    ax.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
    ax.set_title(f"NMS threshold={thresh}\n({len(filtered)} objetos)")
    ax.axis("off")
plt.tight_layout()
plt.show()
# 💭 Reflexión: threshold más bajo → más estricto (menos objetos).
# ¿Por qué? Porque permite menos solapamiento antes de eliminar una caja.
