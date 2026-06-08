import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import supervision as sv
from ultralytics import YOLO
from trackers import ByteTrackTracker
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
urllib.request.urlretrieve(
    "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
    "assets/vehicles.mp4"
)

model = YOLO("yolov8x.pt")
tracker = ByteTrackTracker()
box_annotator = sv.BoxAnnotator()

video_info = sv.VideoInfo.from_video_path("assets/vehicles.mp4")
print(f"Resolución: {video_info.width} × {video_info.height} px")



# Polígono que cubre el carril izquierdo del video
# Los vértices en orden: superior-izq, superior-der, inferior-der, inferior-izq
POLYGON_IZQUIERDO = np.array([
    [0,                         video_info.height // 2],
    [video_info.width // 2,     video_info.height // 2],
    [video_info.width // 2,     video_info.height],
    [0,                         video_info.height],
])

zone = sv.PolygonZone(polygon=POLYGON_IZQUIERDO)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=4)

# Visualizar la zona sobre el primer frame para verificar su posición
cap = cv2.VideoCapture("assets/vehicles.mp4")
ret, primer_frame = cap.read()
cap.release()

frame_con_zona = zone_annotator.annotate(scene=primer_frame.copy())
# plt.figure(figsize=(12, 6))
# plt.imshow(cv2.cvtColor(frame_con_zona, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.title("Zona definida (polígono rojo)")
# plt.show()


label_annotator = sv.LabelAnnotator()
tracker = ByteTrackTracker()

def callback_zona(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame, verbose=False, device='cuda')[0]
    detections = sv.Detections.from_ultralytics(results)
    # El tracker debe ir ANTES de trigger() porque PolygonZone usa tracker_id internamente
    detections = tracker.update(detections)
    
    # trigger() devuelve una máscara booleana (True si el objeto está dentro del polígono)
    # y actualiza zone.current_count automáticamente
    en_zona = zone.trigger(detections=detections)
    detections_en_zona = detections[en_zona]
    
    labels = [f"ID:{tid}" for tid in detections_en_zona.tracker_id]
    
    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections_en_zona)
    annotated = label_annotator.annotate(scene=annotated, detections=detections_en_zona, labels=labels)
    # zone_annotator muestra el polígono y el conteo actual (zone.current_count) automáticamente
    annotated = zone_annotator.annotate(scene=annotated)
    return annotated

# sv.process_video(
#     source_path="assets/vehicles.mp4",
#     target_path="assets/vehicles_zona.mp4",
#     callback=callback_zona
# )
# print("Guardado: assets/vehicles_zona.mp4")



# El polígono es solo una lista de puntos — cambiarla redefine la zona completamente
POLYGON_DERECHO = np.array([
    [video_info.width // 2,     video_info.height // 2],
    [video_info.width,          video_info.height // 2],
    [video_info.width,          video_info.height],
    [video_info.width // 2,     video_info.height],
])

zone_derecho = sv.PolygonZone(polygon=POLYGON_DERECHO)
zone_ann_derecho = sv.PolygonZoneAnnotator(zone=zone_derecho, color=sv.Color.BLUE, thickness=4)

frame_dcho = zone_ann_derecho.annotate(scene=primer_frame.copy())
# plt.figure(figsize=(12, 6))
# plt.imshow(cv2.cvtColor(frame_dcho, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.title("Zona carril derecho (azul)")
# plt.show()
# 💭 Reflexión: ¿Detectarías más vehículos en el carril derecho o en el izquierdo?
# Ejecuta un video con zone_derecho para comprobarlo.


# Procesamos manualmente el primer frame para ver qué devuelve trigger()
tracker = ByteTrackTracker()
results_test = model(primer_frame, verbose=False, device='cuda')[0]
det_test = sv.Detections.from_ultralytics(results_test)
det_test = tracker.update(det_test)

mascara_zona = zone.trigger(detections=det_test)

print(f"Objetos totales en el frame: {len(det_test)}")
print(f"Máscara de zona (True = dentro): {mascara_zona}")
print(f"Objetos dentro de la zona: {mascara_zona.sum()}")
# zone.current_count se actualiza automáticamente con cada llamada a trigger()
print(f"zone.current_count: {zone.current_count}")
# 💭 Reflexión: ¿current_count coincide con mascara_zona.sum()?
# Sí — ambos miden lo mismo pero de formas distintas.



# Línea horizontal a mitad del video
line_start = sv.Point(x=0,                   y=video_info.height // 2)
line_end   = sv.Point(x=video_info.width,    y=video_info.height // 2)

line_zone = sv.LineZone(start=line_start, end=line_end)
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_scale=1.5,
    custom_in_text="Cruces ↓",
    custom_out_text="Cruces ↑"
)

tracker = ByteTrackTracker()

def callback_linea(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame, verbose=False, device='cuda')[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update(detections)
    
    # trigger() actualiza line_zone.in_count y line_zone.out_count internamente
    line_zone.trigger(detections=detections)
    
    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    # Nota: LineZoneAnnotator usa frame= y line_counter= (no scene= como los otros annotators)
    annotated = line_zone_annotator.annotate(frame=annotated, line_counter=line_zone)
    return annotated

# sv.process_video(
#     source_path="assets/vehicles.mp4",
#     target_path="assets/vehicles_linea.mp4",
#     callback=callback_linea
# )
print(f"Guardado: assets/vehicles_linea.mp4")
print(f"Cruces hacia abajo: {line_zone.in_count}")
print(f"Cruces hacia arriba: {line_zone.out_count}")
# 💭 Reflexión: ¿Por qué LineZone cuenta cruces acumulados y no presencia?
# Porque está diseñado para medir flujo (cuántas personas/vehículos pasaron),
# no ocupación (cuántos hay en este momento).
# 💭 Reflexión: ¿En qué casos preferirías LineZone sobre PolygonZone?
# Ejemplos: un peaje (cruces acumulados), un puesto de control, la entrada de un edificio.
# PolygonZone responde "¿cuántos hay AHORA?"; LineZone responde "¿cuántos pasaron EN TOTAL?"



tracker = ByteTrackTracker()
zone.trigger(detections=sv.Detections.empty()) 

def callback_combinado(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame, verbose=False, device='cuda')[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update(detections)

    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = zone_annotator.annotate(scene=annotated)
    annotated = line_zone_annotator.annotate(frame=annotated, line_counter=line_zone)
    return annotated

sv.process_video(
    source_path="assets/vehicles.mp4",
    target_path="assets/vehicles_linea_polygon.mp4",
    callback=callback_combinado
)
