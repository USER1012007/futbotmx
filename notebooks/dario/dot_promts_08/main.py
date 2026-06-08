import supervision as sv
from ultralytics import YOLO, SAM
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path

Path("assets").mkdir(exist_ok=True)
urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg",    "assets/bus.jpg")
urllib.request.urlretrieve("https://ultralytics.com/images/zidane.jpg", "assets/zidane.jpg")

image = cv2.imread("assets/bus.jpg")
print(f"Imagen cargada: {image.shape}")

#######################################################################

yolo_model = YOLO("../yolov8x.pt")
yolo_results = yolo_model(image)[0]
yolo_det     = sv.Detections.from_ultralytics(yolo_results)

# Centro del primer objeto detectado
x1, y1, x2, y2 = yolo_det.xyxy[0]
punto = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
print(f"Primer objeto: clase {yolo_det.class_id[0]}, centro en {punto}")

# Marcar el punto en la imagen para visualizarlo
imagen_con_punto = image.copy()
cv2.circle(imagen_con_punto, punto, radius=10, color=(0, 0, 255), thickness=-1)
plt.figure(figsize=(12, 7))
plt.imshow(cv2.cvtColor(imagen_con_punto, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Punto de partida (centro del primer objeto YOLO)")
plt.show()

#######################################################################

sam_model = SAM("../sam3.pt")

resultados = sam_model.predict(source=image, points=[punto], labels=[1])[0]
detections = sv.Detections.from_ultralytics(resultados)

print(f"Máscaras generadas: {len(detections)}")

annotated = sv.MaskAnnotator(opacity=0.6).annotate(scene=image.copy(), detections=detections)
cv2.circle(annotated, punto, radius=10, color=(0, 0, 255), thickness=-1)
plt.figure(figsize=(12, 7))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("SAM 3 con un punto positivo (punto rojo)")
plt.show()

#######################################################################

print(f"Número de máscaras: {len(detections)}")
if detections.mask is not None:
    for i in range(len(detections)):
        area = int(detections.mask[i].sum())
        conf = detections.confidence[i] if detections.confidence is not None else 0
        print(f"  Máscara {i}: area={area:,} px2, confianza={conf:.3f}")

#######################################################################

# Tomamos tres puntos dentro del primer objeto detectado
puntos_1 = [punto]
puntos_2 = [punto, [int(x1 + (x2-x1)*0.25), int(y1 + (y2-y1)*0.5)]]
puntos_3 = [punto, [int(x1 + (x2-x1)*0.25), int(y1 + (y2-y1)*0.5)],
                   [int(x1 + (x2-x1)*0.75), int(y1 + (y2-y1)*0.5)]]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, pts, n in zip(axes, [puntos_1, puntos_2, puntos_3], [1, 2, 3]):
    res = sam_model.predict(source=image, points=pts, labels=[1]*len(pts))[0]
    det = sv.Detections.from_ultralytics(res)
    scn = sv.MaskAnnotator(opacity=0.6).annotate(scene=image.copy(), detections=det)
    for p in pts:
        cv2.circle(scn, p, 8, (0, 0, 255), -1)
    ax.imshow(cv2.cvtColor(scn, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{n} punto(s)")
    ax.axis("off")
plt.tight_layout()
plt.show()
# 💭 Reflexión: ¿Cambia significativamente la máscara con más puntos?
# En objetos con forma clara, un punto suele bastar.
# Los puntos adicionales ayudan cuando el objeto tiene partes separadas visualmente.

#######################################################################

# Punto positivo: centro del objeto
# Punto negativo: zona de fondo (arriba a la izquierda del objeto)
punto_pos = punto
punto_neg = [max(0, punto[0] - 60), max(0, punto[1] - 80)]

res_neg = sam_model.predict(
    source=image,
    points=[punto_pos, punto_neg],
    labels=[1, 0]   # 1=incluir, 0=excluir
)[0]
det_neg = sv.Detections.from_ultralytics(res_neg)

res_pos = sam_model.predict(source=image, points=[punto_pos], labels=[1])[0]
det_pos = sv.Detections.from_ultralytics(res_pos)

scn_pos = sv.MaskAnnotator(opacity=0.6).annotate(scene=image.copy(), detections=det_pos)
scn_neg = sv.MaskAnnotator(opacity=0.6).annotate(scene=image.copy(), detections=det_neg)
cv2.circle(scn_neg, punto_pos, 8, (0, 255, 0), -1)   # verde = positivo
cv2.circle(scn_neg, punto_neg, 8, (0, 0, 255), -1)   # rojo  = negativo

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
ax1.imshow(cv2.cvtColor(scn_pos, cv2.COLOR_BGR2RGB)); ax1.set_title("Solo punto positivo");              ax1.axis("off")
ax2.imshow(cv2.cvtColor(scn_neg, cv2.COLOR_BGR2RGB)); ax2.set_title("Positivo + negativo (rojo=excluir)"); ax2.axis("off")
plt.tight_layout()
plt.show()
# 💭 Reflexión: ¿El punto negativo redujo el área de la máscara?
# El punto negativo le dice a SAM: "ese píxel pertenece al fondo, no al objeto".

#######################################################################

from ultralytics.models.sam import SAM3SemanticPredictor

# Texto
predictor = SAM3SemanticPredictor(overrides=dict(conf=0.25, task="segment", mode="predict", model="../sam3.pt"))
predictor.set_image(image)
det_texto = sv.Detections.from_ultralytics(predictor(text=["person"])[0])

# Punto
det_punto = sv.Detections.from_ultralytics(
    sam_model.predict(source=image, points=[punto], labels=[1])[0]
)

# Bbox (YOLO)
solo_person = yolo_det[yolo_det.class_id == 0]
det_bbox = sv.Detections.from_ultralytics(
    sam_model(image, bboxes=solo_person.xyxy.tolist())[0]
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, det, titulo in zip(axes,
                           [det_texto, det_punto, det_bbox],
                           ["Texto", "Punto", "Bbox (YOLO)"]):
    scn = sv.MaskAnnotator(opacity=0.6).annotate(scene=image.copy(), detections=det)
    ax.imshow(cv2.cvtColor(scn, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{titulo} — {len(det)} obj.")
    ax.axis("off")
plt.suptitle("Texto vs. Punto vs. Bbox — tres formas de guiar SAM 3", fontsize=13)
plt.tight_layout()
plt.show()
# 💭 Reflexión: ¿Cuál produce la máscara más completa? ¿Cuál la más precisa?
# Texto: mayor cobertura, menos control individual.
# Punto: control directo sobre un objeto, sencillo pero ambiguo en escenas densas.
# Bbox: mayor precisión, requiere detector previo.

#######################################################################

from ultralytics.models.sam import SAM3SemanticPredictor

# Paso 1: texto para descubrir — SAM 3 localiza automáticamente los objetos
predictor_txt = SAM3SemanticPredictor(overrides=dict(
    conf=0.25, task="segment", mode="predict", model="../sam3.pt"
))
predictor_txt.set_image(image)
res_txt  = predictor_txt(text=["person"])[0]
det_txt  = sv.Detections.from_ultralytics(res_txt)
print(f"Texto encontró {len(det_txt)} persona(s)")

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

#######################################################################

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
    ax.imshow(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
    ax.set_title(f"Objeto {i} — clase {yolo_det.class_id[i]}")
    ax.axis("off")
plt.tight_layout()
plt.show()
