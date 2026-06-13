import supervision as sv
from ultralytics.models.sam import SAM3SemanticPredictor
from ultralytics import YOLO, SAM
import torch
import cv2
import matplotlib.pyplot as plt
import urllib.request
from pathlib import Path

Path("assets").mkdir(exist_ok=True)
urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg",    "assets/bus.jpg")
urllib.request.urlretrieve("https://ultralytics.com/images/zidane.jpg", "assets/zidane.jpg")

image = cv2.imread("assets/bus.jpg")
print(f"Imagen cargada: {image.shape}")

#######################################################################

overrides = dict(conf=0.25, task="segment", mode="predict", model="../sam3.pt")
if torch.cuda.is_available():
    overrides["half"] = True   # FP16 solo en GPU

predictor = SAM3SemanticPredictor(overrides=overrides)
print("Predictor listo")

#######################################################################

predictor.set_image(image)

resultados = predictor(text=["person"])[0]
detections = sv.Detections.from_ultralytics(resultados)

print(f"Objetos encontrados: {len(detections)}")
print(f"¿Tiene máscaras?    {detections.mask is not None}")

#######################################################################

box_annotator  = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator(opacity=0.6)

annotated = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated = box_annotator.annotate(scene=annotated,    detections=detections)

plt.figure(figsize=(12, 7))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title('SAM 3 con prompt de texto: "person"')
plt.show()

#######################################################################

if detections.confidence is not None:
    for i, conf in enumerate(detections.confidence):
        area = int(detections.mask[i].sum()) if detections.mask is not None else 0
        print(f"Objeto {i}: confianza={conf:.3f}  area={area:,} px2")

#######################################################################

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, concepto in zip(axes, ["vehicle", "bus", "person"]):
    res = predictor(text=[concepto])[0]
    det = sv.Detections.from_ultralytics(res)
    scn = sv.MaskAnnotator(opacity=0.6).annotate(scene=image.copy(), detections=det)
    ax.imshow(cv2.cvtColor(scn, cv2.COLOR_BGR2RGB))
    ax.set_title(f'"{concepto}" — {len(det)} objetos')
    ax.axis("off")
plt.tight_layout()
plt.show()
# 💭 Reflexión: ¿"vehicle" incluye el bus? ¿Qué ocurre con conceptos ambiguos?
# Conceptos amplios capturan más instancias pero pueden incluir falsos positivos.

#######################################################################

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, umbral in zip(axes, [0.1, 0.3, 0.6]):
    predictor.args.conf = umbral
    res = predictor(text=["person"])[0]
    det = sv.Detections.from_ultralytics(res)
    scn = sv.MaskAnnotator(opacity=0.6).annotate(scene=image.copy(), detections=det)
    ax.imshow(cv2.cvtColor(scn, cv2.COLOR_BGR2RGB))
    ax.set_title(f"conf={umbral}  ({len(det)} objetos)")
    ax.axis("off")
plt.tight_layout()
plt.show()
predictor.args.conf = 0.25   # restaurar valor original
# 💭 Reflexión: ¿Qué umbral produce el mejor resultado visual?
# Un umbral bajo puede incluir detecciones parciales en los bordes de la imagen.

#######################################################################

predictor.args.conf = 0.25
res_texto   = predictor(text=["person"])[0]
det_texto   = sv.Detections.from_ultralytics(res_texto)

yolo_model  = YOLO("../yolov8n.pt")
yolo_r      = yolo_model(image)[0]
yolo_det    = sv.Detections.from_ultralytics(yolo_r)
solo_person = yolo_det[yolo_det.class_id == 0]
sam_bbox    = SAM("../sam3.pt")
res_bbox    = sam_bbox(image, bboxes=solo_person.xyxy.tolist())[0]
det_bbox    = sv.Detections.from_ultralytics(res_bbox)

scn_texto = sv.MaskAnnotator(opacity=0.6).annotate(scene=image.copy(), detections=det_texto)
scn_bbox  = sv.MaskAnnotator(opacity=0.6).annotate(scene=image.copy(), detections=det_bbox)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
ax1.imshow(cv2.cvtColor(scn_texto, cv2.COLOR_BGR2RGB))
ax1.set_title(f'Texto: "person" ({len(det_texto)} objetos)')
ax1.axis("off")
ax2.imshow(cv2.cvtColor(scn_bbox, cv2.COLOR_BGR2RGB))
ax2.set_title(f"Bbox YOLO ({len(det_bbox)} objetos)")
ax2.axis("off")
plt.suptitle("Texto vs. bounding box", fontsize=13)
plt.tight_layout()
plt.show()
# 💭 Reflexión: ¿Cuál detecta más personas? ¿Las máscaras son igual de precisas?
# El texto puede encontrar personas que YOLO no detectó (mayor cobertura).
# Las cajas de YOLO tienden a producir máscaras más precisas en contornos.

#######################################################################

# Escribe tu solución aquí
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, concepto in zip(axes, ["person", "bus", "wheel"]):
    res = predictor(text=[concepto])[0]
    det = sv.Detections.from_ultralytics(res)
    if det.mask is not None and len(det) > 0:
        areas   = det.mask.sum(axis=(1, 2))   # área en píxeles de cada máscara
        idx_max = int(areas.argmax())          # índice del objeto más grande
        mayor   = det[idx_max]                 # sv.Detections con 1 objeto
        annotated = mask_annotator.annotate(scene=image.copy(), detections=mayor)
        ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    # ax.imshow(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
    ax.set_title(f'"{concepto}"')
    ax.axis("off")

plt.tight_layout()
plt.show()
