import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import supervision as sv
from ultralytics import SAM
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path
import torch
import gc

gc.collect()
torch.cuda.empty_cache()

Path("assets").mkdir(exist_ok=True)
urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", "assets/bus.jpg")
image = cv2.imread("assets/bus.jpg")

# model = SAM("sam3.pt")
model = YOLO("yolov8x.pt")
results = model(image, half=True, device='cuda')[0]
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



# La comparación crea una máscara booleana (array de True/False) para cada detección
# sv.Detections[mascara_booleana] devuelve solo las filas donde la máscara es True
# mascara = detections.confidence > 0.5
# alta_confianza = detections[mascara]
#
# print(f"Total: {len(detections)} | Con confianza > 0.5: {len(alta_confianza)}")
# mostrar(alta_confianza, "Solo confianza > 0.5")
#

# Clases disponibles en esta imagen (COCO dataset)
print("Objetos detectados:")
for class_id in sorted(set(detections.class_id)):
    n = (detections.class_id == class_id).sum()
    print(f"  Clase {class_id} ({results.names[class_id]}): {n} detecciones")

# Clase 0 = 'person' en COCO
personas = detections[detections.class_id == 0]
print(f"\nSolo personas: {len(personas)}")
# mostrar(personas, "Solo personas (clase 0)")



# Para combinar condiciones, DEBES usar & (AND elemento a elemento), no 'and'
# 'and' en Python es para valores booleanos individuales, no arrays
personas_seguras = detections[
    (detections.class_id == 0) & (detections.confidence > 0.6)
]
print(f"Personas con confianza > 60%: {len(personas_seguras)}")
# mostrar(personas_seguras, "Personas con confianza > 60%")



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



# detections.area calcula automáticamente el área de cada bounding box en píxeles²
areas = detections.area
print(f"Área mínima:  {areas.min():.0f} px²")
print(f"Área máxima:  {areas.max():.0f} px²")
print(f"Área promedio: {areas.mean():.0f} px²")

# Objetos muy pequeños pueden ser falsos positivos o ruido
objetos_grandes = detections[detections.area > 8000]
print(f"\nObjetos con área > 5000 px²: {len(objetos_grandes)}")
# mostrar(objetos_grandes, "Solo objetos grandes (área > 5000 px²)")



# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# for ax, thresh in zip(axes, [0.3, 0.5, 0.8]):
#     filtered = sv.Detections.merge([det_baja, det_alta]).with_nms(threshold=thresh)
#     etiquetas = [results.names[c] for c in filtered.class_id]
#     scene = box_annotator.annotate(scene=image.copy(), detections=filtered)
#     scene = label_annotator.annotate(scene=scene, detections=filtered, labels=etiquetas)
#     ax.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
#     ax.set_title(f"NMS threshold={thresh}\n({len(filtered)} objetos)")
#     ax.axis("off")
# plt.tight_layout()
# plt.show()

# 💭 Reflexión: threshold más bajo → más estricto (menos objetos).
# ¿Por qué? Porque permite menos solapamiento antes de eliminar una caja.


# Clase 5 = 'bus' en COCO
# != excluye en lugar de incluir — la lógica es idéntica a la inclusión
sin_buses = detections[detections.class_id != 5]
print(f"Con buses: {len(detections)} | Sin buses: {len(sin_buses)}")
# mostrar(sin_buses, "Sin autobuses (clase 5 excluida)")
# 💭 Reflexión: ¿Qué otras clases podrías excluir o filtrar?
# Prueba cambiar el 5 por otro class_id de la tabla de arriba.



# np.argsort devuelve los índices que ordenarían el array de menor a mayor
# [::-1] invierte el orden (de mayor a menor)
# [:3] toma los primeros 3 índices
indices_top3 = np.argsort(detections.confidence)[::-1][:3]
top3 = detections[indices_top3]

print("Top 3 detecciones por confianza:")
for i in range(len(top3)):
    print(f"  {results.names[top3.class_id[i]]}: {top3.confidence[i]:.1%}")

# mostrar(top3, "Top 3 detecciones más confiables")
# 💭 Reflexión: ¿Son siempre los objetos más grandes los más confiables?
# No necesariamente — depende del entrenamiento del modelo y la imagen.



centros_x = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2
mitad_imagen = image.shape[1] / 2

sin_buses = detections[centros_x > mitad_imagen]
print(f"mitad derecha: {len(sin_buses)}")
mostrar(sin_buses, "mitad derecha")
