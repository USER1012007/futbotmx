import os
# Evita fragmentación antes de que PyTorch inicialice el contexto de CUDA
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import gc
import torch
import numpy as np
import urllib.request
from pathlib import Path
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics import YOLO, SAM
from ultralytics.models.sam import SAM3SemanticPredictor

# Limpieza inicial
gc.collect()
torch.cuda.empty_cache()

# Descarga de assets
Path("assets").mkdir(exist_ok=True)
if not Path("assets/bus.jpg").exists():
    urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", "assets/bus.jpg")

image = cv2.imread("assets/bus.jpg")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
half_precision = True if device == 'cuda' else False

# --- PASO 1: DETECCIÓN CON YOLO ---
# Nota: Si sigue fallando por VRAM, cambia "yolov8x.pt" por "yolov8m.pt"
yolo_model = YOLO("yolov8x.pt")
yolo_results = yolo_model(image, device=device, half=half_precision)[0]
yolo_det = sv.Detections.from_ultralytics(yolo_results)

x1, y1, x2, y2 = yolo_det.xyxy[0]
punto = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
print(f"YOLO -> Primer objeto: clase {yolo_det.class_id[0]}, centro en {punto}")

# Matar a YOLO para revivir la VRAM
del yolo_model
del yolo_results
gc.collect()
torch.cuda.empty_cache()

# --- PASO 2: SEGMENTACIÓN POR PUNTOS CON SAM ---
sam_model = SAM("sam3.pt")

# Puntos de prueba
puntos_3 = [punto, 
            [int(x1 + (x2-x1)*0.25), int(y1 + (y2-y1)*0.5)],
            [int(x1 + (x2-x1)*0.75), int(y1 + (y2-y1)*0.5)]]

# Inferencia con SAM usando FP16 (half=True)
resultados = sam_model.predict(source=image, points=puntos_3, labels=[1]*len(puntos_3), device=device, half=half_precision)[0]
detections = sv.Detections.from_ultralytics(resultados)
print(f"SAM Puntos -> Máscaras generadas: {len(detections)}")

# Matar a SAM temporalmente antes de pasar al predictor semántico por texto
# del sam_model
# del resultados
gc.collect()
torch.cuda.empty_cache()

# --- PASO 3: FLUJO TEXTO -> PUNTO (SAM3 Semántico) ---
# Usamos half=True en los overrides del predictor para cuidar tus 6GB de VRAM
predictor_txt = SAM3SemanticPredictor(overrides=dict(
    conf=0.25, task="segment", mode="predict", model="sam3.pt", device=device, half=half_precision
))
predictor_txt.set_image(image)
res_txt = predictor_txt(text=["person"])[0]
det_txt = sv.Detections.from_ultralytics(res_txt)
print(f"SAM Texto -> Encontró {len(det_txt)} persona(s)")



# Paso 2: punto para precisar — elegimos la primera persona detectada por texto
# y la segmentamos con un punto para tener control individual
if len(det_txt) > 0:
    x1, y1, x2, y2 = det_txt.xyxy[0]
    centro = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
    print(f"Centro de la primera persona (vía texto): {centro}")

    res_punto  = sam_model.predict(source=image, points=[centro], labels=[1])[0]
    det_punto  = sv.Detections.from_ultralytics(res_punto)

    scn_txt   = sv.MaskAnnotator(opacity=0.4).annotate(scene=image.copy(), detections=det_txt)
    scn_punto = sv.MaskAnnotator(opacity=0.7).annotate(scene=image.copy(), detections=det_punto)
    cv2.circle(scn_punto, centro, 10, (0, 0, 255), -1)  # punto rojo

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    ax1.imshow(cv2.cvtColor(scn_txt,   cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Texto "person": {len(det_txt)} objetos (opacidad baja)')
    ax1.axis("off")
    ax2.imshow(cv2.cvtColor(scn_punto, cv2.COLOR_BGR2RGB))
    ax2.set_title("Punto (coord. del texto): 1 objeto preciso (punto rojo)")
    ax2.axis("off")
    plt.suptitle("Flujo: texto para descubrir → punto para precisar", fontsize=13)
    plt.tight_layout()
    plt.show()
    # 💭 Reflexión: ¿Cuándo es útil esta combinación?
    # Texto te da la ubicación sin saber las coordenadas de antemano.
    # Punto te da control individual sobre qué instancia segmentar.

# Liberación final
del predictor_txt
gc.collect()
torch.cuda.empty_cache()

# Escribe tu solución aquí
mask_annotator_reto = sv.MaskAnnotator(opacity=0.6)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, ax in enumerate(axes):
    if i >= len(yolo_det):
        ax.axis("off")
        continue
    for i in range(min(3, len(yolo_det))):
        x1, y1, x2, y2 = yolo_det.xyxy[i]
        centro = [int((x1+x2)/2), int((y1+y2)/2)]
        res = sam_model.predict(source=image, points=[centro], labels=[1], verbose=False)[0]
        det = sv.Detections.from_ultralytics(res)
    annotated = mask_annotator_reto.annotate(scene=image.copy(), detections=det)
    ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Objeto {i} — clase {yolo_det.class_id[i]}")
    ax.axis("off")
plt.tight_layout()
plt.show()

gc.collect()
torch.cuda.empty_cache()
