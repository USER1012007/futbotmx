import supervision as sv
from ultralytics import YOLO, SAM
from trackers import ByteTrackTracker
import cv2
import numpy as np
import urllib.request
from pathlib import Path
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc

gc.collect()
torch.cuda.empty_cache()

Path("assets").mkdir(exist_ok=True)
urllib.request.urlretrieve(
    "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
    "assets/vehicles.mp4"
)

yolo_model = YOLO("yolov8n.pt")
sam_model  = SAM("sam3.pt")
tracker    = ByteTrackTracker()

video_info = sv.VideoInfo.from_video_path("assets/vehicles.mp4")
print(f"Resolución: {video_info.width} × {video_info.height}")
print(f"FPS: {video_info.fps} | Total frames: {video_info.total_frames}")


# pipeline A, IMPORTANTE

mask_annotator  = sv.MaskAnnotator(opacity=0.6)
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

tracker = ByteTrackTracker()

def procesar_frame_sam(frame: np.ndarray, _: int) -> np.ndarray:
    # Paso 1: Detectar con YOLO y asignar IDs persistentes
    yolo_results = yolo_model(frame, verbose=False)[0]
    yolo_det     = sv.Detections.from_ultralytics(yolo_results)
    yolo_det     = tracker.update(yolo_det)
    
    if len(yolo_det) == 0:
        return frame
    
    # Paso 2: SAM genera máscaras usando las cajas de YOLO como guía
    # .tolist() es necesario porque SAM espera lista de Python, no array NumPy
    bboxes      = yolo_det.xyxy.tolist()
    sam_results = sam_model(frame, bboxes=bboxes, verbose=False)[0]
    sam_det     = sv.Detections.from_ultralytics(sam_results)
    
    # Paso 3: Transferir atributos de YOLO a SAM
    # SAM preserva el orden de los bboxes de entrada — la alineación posicional es segura
    if len(sam_det) == len(yolo_det):
        sam_det.tracker_id = yolo_det.tracker_id
        sam_det.class_id   = yolo_det.class_id
        sam_det.confidence = yolo_det.confidence
    
    labels = [
        f"ID:{tid} {yolo_results.names[c]}"
        for tid, c in zip(sam_det.tracker_id, sam_det.class_id)
    ]
    
    annotated = mask_annotator.annotate(scene=frame.copy(), detections=sam_det)
    annotated = label_annotator.annotate(scene=annotated, detections=sam_det, labels=labels)
    annotated = trace_annotator.annotate(scene=annotated, detections=sam_det)
    return annotated

# sv.process_video(
#     source_path="assets/vehicles.mp4",
#     target_path="assets/vehicles_sam.mp4",
#     callback=procesar_frame_sam
# )
#
print("Guardado: assets/vehicles_sam.mp4")

# Inspeccionamos un frame para ver qué contiene sam_det ANTES y DESPUÉS de la transferencia
# tracker = ByteTrackTracker()
# cap = cv2.VideoCapture("assets/vehicles.mp4")
# ret, frame_test = cap.read()
# cap.release()
#
# yolo_r   = yolo_model(frame_test, verbose=False)[0]
# yolo_det = sv.Detections.from_ultralytics(yolo_r)
# yolo_det = tracker.update(yolo_det)
#
# bboxes   = yolo_det.xyxy.tolist()
# sam_r    = sam_model(frame_test, bboxes=bboxes, verbose=False)[0]
# sam_det  = sv.Detections.from_ultralytics(sam_r)
#
# print("SAM ANTES de transferencia:")
# print(f"  tracker_id: {sam_det.tracker_id}")   # None — SAM no sabe de tracking
# print(f"  class_id:   {sam_det.class_id}")      # None o índice SAM interno
# print(f"  mask shape: {sam_det.mask.shape if sam_det.mask is not None else None}")
#
# if len(sam_det) == len(yolo_det):
#     sam_det.tracker_id = yolo_det.tracker_id
#     sam_det.class_id   = yolo_det.class_id
#     sam_det.confidence = yolo_det.confidence
#
# print("\nSAM DESPUÉS de transferencia:")
# print(f"  tracker_id: {sam_det.tracker_id}")   # ahora tiene IDs de ByteTrack
# print(f"  class_id:   {sam_det.class_id}")      # ahora tiene clases de YOLO



# Combinamos PolygonZone (NB05) con SAM (NB06/NB07)
# Solo segmentamos los objetos que están dentro de la zona
# POLYGON = np.array([
#     [0,                         video_info.height // 2],
#     [video_info.width // 2,     video_info.height // 2],
#     [video_info.width // 2,     video_info.height],
#     [0,                         video_info.height],
# ])
# zone     = sv.PolygonZone(polygon=POLYGON)
# zone_ann = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED, thickness=3)
#
# tracker = ByteTrackTracker()
#
# def callback_zona_sam(frame: np.ndarray, _: int) -> np.ndarray:
#     yolo_r   = yolo_model(frame, verbose=False)[0]
#     yolo_det = sv.Detections.from_ultralytics(yolo_r)
#     yolo_det = tracker.update(yolo_det)
#
#     # Filtrar solo objetos en la zona ANTES de llamar a SAM
#     # — SAM es lento: no gastar tiempo en objetos fuera de interés
#     en_zona  = zone.trigger(detections=yolo_det)
#     yolo_det = yolo_det[en_zona]
#
#     annotated = frame.copy()
#     if len(yolo_det) > 0:
#         bboxes   = yolo_det.xyxy.tolist()
#         sam_r    = sam_model(frame, bboxes=bboxes, verbose=False)[0]
#         sam_det  = sv.Detections.from_ultralytics(sam_r)
#         if len(sam_det) == len(yolo_det):
#             sam_det.tracker_id = yolo_det.tracker_id
#             sam_det.class_id   = yolo_det.class_id
#         annotated = mask_annotator.annotate(scene=annotated, detections=sam_det)
#
#     annotated = zone_ann.annotate(scene=annotated)
#     return annotated
#
# sv.process_video(
#     source_path="assets/vehicles.mp4",
#     target_path="assets/vehicles_sam_zona.mp4",
#     callback=callback_zona_sam
# )
# print("Guardado: assets/vehicles_sam_zona.mp4")
# 💭 Reflexión: ¿Cuánto más rápido es este callback comparado con el que segmenta todo?
# Al filtrar antes de SAM, reducimos el número de objetos que SAM procesa por frame.




# MaskAnnotator tiene una opacidad fija para todos los objetos.
# Para opacidad por objeto, podemos anotar objeto por objeto manualmente.
# tracker = ByteTrackTracker()
#
# def callback_opacidad(frame: np.ndarray, _: int) -> np.ndarray:
#     yolo_r   = yolo_model(frame, verbose=False)[0]
#     yolo_det = sv.Detections.from_ultralytics(yolo_r)
#     yolo_det = tracker.update(yolo_det)
#
#     if len(yolo_det) == 0:
#         return frame
#
#     bboxes   = yolo_det.xyxy.tolist()
#     sam_r    = sam_model(frame, bboxes=bboxes, verbose=False)[0]
#     sam_det  = sv.Detections.from_ultralytics(sam_r)
#     if len(sam_det) == len(yolo_det):
#         sam_det.tracker_id = yolo_det.tracker_id
#         sam_det.confidence = yolo_det.confidence
#
#     annotated = frame.copy()
#     # Anotar objeto por objeto usando su confianza como opacidad
#     for i in range(len(sam_det)):
#         det_i    = sam_det[i]
#         # Alta confianza → máscara más opaca (más "seguro" el objeto)
#         opacidad = float(det_i.confidence[0]) if det_i.confidence is not None else 0.5
#         annotated = sv.MaskAnnotator(opacity=opacidad).annotate(scene=annotated, detections=det_i)
#     return annotated
#
# sv.process_video(
#     source_path="assets/vehicles.mp4",
#     target_path="assets/vehicles_sam_opacidad.mp4",
#     callback=callback_opacidad
# )
# print("Guardado: assets/vehicles_sam_opacidad.mp4")
# 💭 Reflexión: ¿Qué objetos se ven con máscara más opaca?
# Los que el modelo detecta con más certeza — generalmente los más grandes y visibles.



# Rastreamos cómo cambia el área de la máscara de cada objeto frame a frame
# — útil para detectar cuándo un objeto se acerca o aleja de la cámara
areas_por_id = {}   # {tracker_id: [área_frame_0, área_frame_1, ...]}

tracker = ByteTrackTracker()

def callback_areas(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    yolo_r   = yolo_model(frame, verbose=False)[0]
    yolo_det = sv.Detections.from_ultralytics(yolo_r)
    yolo_det = tracker.update(yolo_det)
    
    if len(yolo_det) == 0:
        return frame
    
    bboxes   = yolo_det.xyxy.tolist()
    sam_r    = sam_model(frame, bboxes=bboxes, verbose=False)[0]
    sam_det  = sv.Detections.from_ultralytics(sam_r)
    if len(sam_det) == len(yolo_det):
        sam_det.tracker_id = yolo_det.tracker_id
    
    # Registrar el área de cada objeto en este frame
    if sam_det.mask is not None:
        for i in range(len(sam_det)):
            tid  = sam_det.tracker_id[i]
            area = int(sam_det.mask[i].sum())  # número de píxeles True
            if tid not in areas_por_id:
                areas_por_id[tid] = []
            areas_por_id[tid].append(area)
    
    return mask_annotator.annotate(scene=frame.copy(), detections=sam_det)

sv.process_video(
    source_path="assets/vehicles.mp4",
    target_path="assets/vehicles_sam_areas.mp4",
    callback=callback_areas
)

# Mostrar evolución del área de los 3 objetos con más frames registrados
import matplotlib.pyplot as plt
ids_con_datos = sorted(areas_por_id, key=lambda k: len(areas_por_id[k]), reverse=True)[:3]
plt.figure(figsize=(12, 4))
for tid in ids_con_datos:
    plt.plot(areas_por_id[tid], label=f"ID {tid}")
plt.xlabel("Frame")
plt.ylabel("Área de máscara (px²)")
plt.title("Evolución del área de máscara por objeto")
plt.legend()
plt.tight_layout()
plt.show()
# 💭 Reflexión: ¿Qué evento físico causa que el área aumente o disminuya?
# Un aumento sostenido → el objeto se acerca a la cámara.
# Una disminución → se aleja. Un cambio brusco → oclusión parcial o nueva detección.
