import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
import urllib.request
from pathlib import Path

Path("assets").mkdir(exist_ok=True)

print("Descargando video de muestra (puede tardar un momento)...")
urllib.request.urlretrieve(
    "https://media.roboflow.com/supervision/video-examples/vehicles.mp4",
    "assets/vehicles.mp4"
)
print("Video listo.")

video_info = sv.VideoInfo.from_video_path("assets/vehicles.mp4")
print(f"\nResolución: {video_info.width} × {video_info.height} px")
print(f"FPS: {video_info.fps}")
print(f"Duración: {video_info.total_frames / video_info.fps:.1f} segundos")

#######################################################################

model = YOLO("../yolov8x.pt")

# sv.ByteTrack es la API clásica de Supervision — funciona pero está siendo reemplazada.
# En versiones futuras usarás ByteTrackTracker del paquete 'trackers'.
# Para este tutorial, sv.ByteTrack es suficiente y más directo.
tracker = sv.ByteTrack()

# Inspeccionamos un solo frame para ver el estado ANTES del tracker
cap = cv2.VideoCapture("assets/vehicles.mp4")
ret, primer_frame = cap.read()
cap.release()

results_test = model(primer_frame, verbose=False)[0]
det_sin_tracker = sv.Detections.from_ultralytics(results_test)

print("ANTES de pasar por el tracker:")
print(f"  tracker_id: {det_sin_tracker.tracker_id}")
# tracker_id es None porque nadie ha asignado IDs todavía

det_con_tracker = tracker.update_with_detections(det_sin_tracker)
print("\nDESPUÉS de pasar por el tracker:")
print(f"  tracker_id: {det_con_tracker.tracker_id}")
# Ahora cada detección tiene un número de identificación único

#######################################################################

# Reiniciamos el tracker para que los IDs empiecen desde 1
# Sin reset(), los IDs continuarían desde donde quedaron en la celda anterior
tracker.reset()

box_annotator   = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
# TraceAnnotator dibuja la trayectoria de cada objeto — necesita tracker_id para funcionar

def procesar_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # El tracker compara estas detecciones con el frame anterior
    # y asigna el mismo tracker_id si reconoce el objeto
    detections = tracker.update_with_detections(detections)
    
    # Usamos tracker_id (quién es) no class_id (qué tipo de objeto es)
    labels = [f"ID:{tid}" for tid in detections.tracker_id]
    
    annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
    annotated = trace_annotator.annotate(scene=annotated, detections=detections)
    return annotated

# sv.process_video(
#     source_path="assets/vehicles.mp4",
#     target_path="assets/vehicles_tracked.mp4",
#     callback=procesar_frame
# )
print("Video guardado: assets/vehicles_tracked.mp4")

#######################################################################

# Procesamos los primeros 3 frames manualmente para ver cómo evolucionan los IDs
tracker.reset()

cap = cv2.VideoCapture("assets/vehicles.mp4")
for frame_num in range(3):
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, verbose=False)[0]
    det = sv.Detections.from_ultralytics(results)
    det = tracker.update_with_detections(det)
    print(f"Frame {frame_num}: {len(det)} objetos | IDs: {det.tracker_id}")
cap.release()

#######################################################################

tracker.reset()

def callback_clase_id(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    det = sv.Detections.from_ultralytics(results)
    det = tracker.update_with_detections(det)
    # Combinar el nombre de la clase con el ID del tracker
    # — más informativo que mostrar solo el ID
    labels = [
        f"{results.names[c]} #{tid}"
        for c, tid in zip(det.class_id, det.tracker_id)
    ]
    scene = box_annotator.annotate(scene=frame.copy(), detections=det)
    scene = trace_annotator.annotate(scene=frame.copy(), detections=det)
    return label_annotator.annotate(scene=scene, detections=det, labels=labels)

# sv.process_video(
#     source_path="assets/vehicles.mp4",
#     target_path="assets/vehicles_clase_id.mp4",
#     callback=callback_clase_id
# )
print("Guardado: assets/vehicles_clase_id.mp4")
# 💭 Reflexión: ¿Qué label resulta más útil para tu aplicación:
# solo el ID, solo la clase, o ambos?

#######################################################################

# Combinar filtrado (NB03) con tracking — el orden importa:
# filtramos ANTES de pasar al tracker para que no gaste IDs en objetos que no nos interesan
CLASE_OBJETIVO = 2  # 'car' en COCO

tracker.reset()

def callback_solo_autos(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    det = sv.Detections.from_ultralytics(results)
    # Filtrar ANTES del tracker — así el tracker solo gestiona la clase de interés
    det = det[det.class_id == CLASE_OBJETIVO]
    det = tracker.update_with_detections(det)
    labels = [f"auto #{tid}" for tid in det.tracker_id]
    scene = box_annotator.annotate(scene=frame.copy(), detections=det)
    return label_annotator.annotate(scene=scene, detections=det, labels=labels)

# sv.process_video(
#     source_path="assets/vehicles.mp4",
#     target_path="assets/vehicles_autos.mp4",
#     callback=callback_solo_autos
# )
print("Guardado: assets/vehicles_autos.mp4")
# 💭 Reflexión: ¿Por qué conviene filtrar ANTES de pasar al tracker
# en lugar de filtrar DESPUÉS?

#######################################################################

# Escribe tu solución aquí
frame_count = {}
tracker.reset()

def mi_callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    det = sv.Detections.from_ultralytics(results)
    det = tracker.update_with_detections(det)
    
    for tid in det.tracker_id:
        frame_count[tid] = frame_count.get(tid, 0) + 1
    labels = [f"#{tid} ({frame_count[tid]}f)" for tid in det.tracker_id]
    
    scene = box_annotator.annotate(scene=frame.copy(), detections=det)
    
    return label_annotator.annotate(scene=scene, detections=det, labels=labels)

sv.process_video(
    source_path="assets/vehicles.mp4",
    target_path="assets/vehicles_frames_exercise.mp4",
    callback=mi_callback
)
