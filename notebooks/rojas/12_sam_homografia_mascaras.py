import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import supervision as sv
from pathlib import Path

image_bgr = cv2.imread("assets/futbot-01.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

SOURCE_POINTS = np.float32([[20, 165], [590, 40], [592, 1052], [15, 985]])
CAMPO_W, CAMPO_H = 364, 486
ESCALA_PX_CM = 2.0  # 1 cm real = 2 px en el campo canónico (RCJ SF23: 182 × 243 cm)
TARGET_POINTS = np.float32([[0, 0], [CAMPO_W, 0], [CAMPO_W, CAMPO_H], [0, CAMPO_H]])
H = cv2.getPerspectiveTransform(SOURCE_POINTS, TARGET_POINTS)

CLASS_NAMES = {0: "azul", 1: "rojo", 2: "balón"}
COLORS_HEX  = {0: "#00b4d8", 1: "#ef233c", 2: "#ff9500"}
COLORS_BGR  = {0: (216, 180, 0), 1: (60, 35, 239), 2: (0, 149, 255)}

print(f"Imagen: {image_bgr.shape[1]}×{image_bgr.shape[0]} px | H: {H.shape}")



# (copiado de NB11 — sin cambios)
def detect_robots_hsv(frame_bgr: np.ndarray, min_area: int = 40) -> sv.Detections:
    """Detecta robots y balón por color HSV. class_id: 0=azul, 1=rojo, 2=balón."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask_azul  = cv2.inRange(hsv, np.array([95,  60,  50]), np.array([125, 255, 255]))
    mask_rojo  = cv2.inRange(hsv, np.array([  0,  70,  40]), np.array([ 10, 255, 255]))
    mask_rojo |= cv2.inRange(hsv, np.array([165,  70,  40]), np.array([179, 255, 255]))
    mask_balon = cv2.inRange(hsv, np.array([  4, 130, 130]), np.array([ 16, 255, 255]))
    # mask_balon[0:200, :] = 0  # Elimina la portería amarilla de arriba
    # mask_azul[850:, :] = 0    # Elimina la portería azul de abajo
    xyxy_list, class_ids = [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for mask, cid in [(mask_azul, 0), (mask_rojo, 1), (mask_balon, 2)]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area: continue
            x, y, w, h_box = cv2.boundingRect(cnt)
            xyxy_list.append([x, y, x + w, y + h_box])
            class_ids.append(cid)
    if not xyxy_list: return sv.Detections.empty()
    return sv.Detections(xyxy=np.array(xyxy_list, dtype=np.float32),
                         class_id=np.array(class_ids, dtype=int))

dets_hsv = detect_robots_hsv(image_bgr)
print(f"Detecciones HSV: {len(dets_hsv)}  — bboxes que usaremos como prompts para SAM")



from ultralytics import SAM

sam_model = SAM("sam3.pt")
print("✅ SAM cargado")

# Obtener máscaras usando los bboxes del detector HSV como prompts
bboxes = dets_hsv.xyxy.tolist()
results = sam_model(image_bgr, bboxes=bboxes, verbose=False)
dets_sam = sv.Detections.from_ultralytics(results[0])

# Transferir class_id del detector HSV (SAM no conoce las clases)
if len(dets_sam) == len(dets_hsv):
    dets_sam.class_id = dets_hsv.class_id

print(f"Detecciones SAM: {len(dets_sam)}")
print(f"  Máscaras shape: {dets_sam.mask.shape if dets_sam.mask is not None else 'None'}")
print(f"  Dtype: {dets_sam.mask.dtype if dets_sam.mask is not None else 'None'}")



palette   = sv.ColorPalette.from_hex(list(COLORS_HEX.values()))
mask_ann  = sv.MaskAnnotator(color=palette, opacity=0.5)
box_ann   = sv.BoxAnnotator(color=palette, thickness=2)
label_ann = sv.LabelAnnotator(color=palette, text_color=sv.Color.WHITE)

labels = [CLASS_NAMES.get(int(c), "?") for c in (dets_sam.class_id if dets_sam.class_id is not None else [])]
vis = mask_ann.annotate(image_rgb.copy(), dets_sam)
vis = box_ann.annotate(vis, dets_sam)
vis = label_ann.annotate(vis, dets_sam, labels=labels)

plt.figure(figsize=(5.6, 10))
plt.imshow(vis)
plt.title(f"SAM: {len(dets_sam)} máscaras de robots (opacity=0.5)")
plt.axis("off")
plt.tight_layout()
plt.show()



def project_mask_contour(mask: np.ndarray, H: np.ndarray) -> np.ndarray | None:
    """Proyecta el contorno exterior de una máscara al campo canónico."""
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    pts = cnt.reshape(-1, 1, 2).astype(np.float32)
    pts_proj = cv2.perspectiveTransform(pts, H)
    return pts_proj.reshape(-1, 2).astype(np.int32)


def draw_tactical_with_masks(dets_sam: sv.Detections, H: np.ndarray,
                              campo_w: int = CAMPO_W, campo_h: int = CAMPO_H) -> np.ndarray:
    """Dibuja el mapa táctico con rellenos y contornos de máscaras SAM proyectadas."""
    canvas = np.zeros((campo_h, campo_w, 3), dtype=np.uint8)
    canvas[:] = (50, 67, 27)

    # Líneas del campo
    cv2.rectangle(canvas, (0, 0), (campo_w - 1, campo_h - 1), (120, 200, 116), 2)
    cv2.line(canvas, (0, campo_h // 2), (campo_w, campo_h // 2), (120, 200, 116), 1)
    cv2.circle(canvas, (campo_w // 2, campo_h // 2), int(30 * ESCALA_PX_CM), (120, 200, 116), 1)
    pen_w = int(80 * ESCALA_PX_CM); pen_h = int(40 * ESCALA_PX_CM)
    pen_x = (campo_w - pen_w) // 2
    cv2.rectangle(canvas, (pen_x, 0), (pen_x + pen_w, pen_h), (120, 200, 116), 1)
    cv2.rectangle(canvas, (pen_x, campo_h - pen_h),
                           (pen_x + pen_w, campo_h - 1), (120, 200, 116), 1)
    goal_wp = int(60 * ESCALA_PX_CM); goal_xp = (campo_w - goal_wp) // 2
    cv2.rectangle(canvas, (goal_xp, 0), (goal_xp + goal_wp, int(8 * ESCALA_PX_CM)), (0, 214, 255), 2)
    cv2.rectangle(canvas, (goal_xp, campo_h - int(8 * ESCALA_PX_CM)),
                           (goal_xp + goal_wp, campo_h - 1), (216, 180, 0), 2)

    masks     = dets_sam.mask
    class_ids = (dets_sam.class_id if dets_sam.class_id is not None
                 else np.zeros(len(dets_sam), dtype=int))

    if masks is not None:
        for m, cid in zip(masks, class_ids):
            color = COLORS_BGR.get(int(cid), (200, 200, 200))

            # Opción A: relleno semitransparente (warp de la máscara)
            m_uint8  = m.astype(np.uint8) * 255
            m_warped = cv2.warpPerspective(m_uint8, H, (campo_w, campo_h))
            overlay  = canvas.copy()
            overlay[m_warped > 0] = color
            canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)

            # Opción B: contorno proyectado
            contorno = project_mask_contour(m, H)
            if contorno is not None and len(contorno) > 2:
                cv2.polylines(canvas, [contorno], True, color, 2)

            # Centroide de la máscara
            ys, xs = np.where(m)
            if len(xs) > 0:
                cx = int(xs.mean()); cy = int(ys.mean())
                pt = np.float32([[[cx, cy]]])
                proj = cv2.perspectiveTransform(pt, H)
                px, py = int(proj[0][0][0]), int(proj[0][0][1])
                if 0 <= px < campo_w and 0 <= py < campo_h:
                    cv2.circle(canvas, (px, py), 8, color, -1)
                    cv2.circle(canvas, (px, py), 8, (255, 255, 255), 1)

    return canvas


tactical_bgr = draw_tactical_with_masks(dets_sam, H)
plt.figure(figsize=(6, 8.0))
plt.imshow(cv2.cvtColor(tactical_bgr, cv2.COLOR_BGR2RGB))
plt.title("Mapa táctico con máscaras SAM proyectadas — relleno + contorno")
plt.axis("off")
plt.tight_layout()
plt.show()



from ultralytics import SAM
import torch

# Detectar automáticamente si CUDA está disponible, si no, usar CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Ejecutando en: {device.upper()}")

# Cargar el modelo y mandarlo a la GPU
sam_model = SAM("sam3.pt").to(device)
print("✅ SAM cargado en el dispositivo correcto")

OUTPUT_VIDEO = "assets/futbot_nb12.mp4"
SOURCE_VIDEO = "assets/futbot-2s.mp4"

video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO)
video_info.width = 998
video_info.height = 486

def callback_nb12(frame: np.ndarray, _: int) -> np.ndarray:
    escala  = CAMPO_H / frame.shape[0]
    orig_w  = int(frame.shape[1] * escala)   # Típicamente ~634 si es 1080p
    # 1. Detección HSV
    dets = detect_robots_hsv(frame)
    
    # MANEJO DE FRAMES VACÍOS: Forzar que mida exactamente 998 de ancho
    if len(dets) == 0:
        resized = cv2.resize(frame, (orig_w, CAMPO_H))
        
        # El espacio restante para llegar a 998 se lo dividen el mapa cenital y el táctico
        # Si resized es 634, quedan 364 píxeles libres. Los dividimos equitativamente o respetamos CAMPO_W.
        # Para no errar en el redondeo, calculamos el remanente exacto:
        ancho_restante = 998 - orig_w  # Ej: 998 - 634 = 364
        w_cenital = ancho_restante // 2
        w_tactico = ancho_restante - w_cenital
        
        blank_warped   = np.zeros((CAMPO_H, w_cenital, 3), dtype=np.uint8)
        blank_tactical = np.zeros((CAMPO_H, w_tactico, 3), dtype=np.uint8)
        
        # Color de fondo verde cancha para los paneles vacíos
        blank_warped[:] = (50, 67, 27)
        blank_tactical[:] = (50, 67, 27)
        
        return np.hstack([resized, blank_warped, blank_tactical])

    # 2. SAM con bboxes como prompts
    results  = sam_model(frame, bboxes=dets.xyxy.tolist(), verbose=False)
    dets_sam = sv.Detections.from_ultralytics(results[0])
    if len(dets_sam) == len(dets):
        dets_sam.class_id = dets.class_id

    # 3. Vista cenital
    warped = cv2.warpPerspective(frame, H, (CAMPO_W, CAMPO_H))

    # 4. Mapa táctico con máscaras
    tactical = draw_tactical_with_masks(dets_sam, H)
    w_tactical_esperado = 998 - orig_w - CAMPO_W
    if tactical.shape[1] != w_tactical_esperado or tactical.shape[0] != CAMPO_H:
        tactical = cv2.resize(tactical, (w_tactical_esperado, CAMPO_H))
    # 5. Frame original redimensionado con máscara superpuesta
    frame_annotated = frame.copy()
    if dets_sam.mask is not None:
        frame_rgb = cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB)
        frame_rgb = sv.MaskAnnotator(opacity=0.4).annotate(frame_rgb, dets_sam)
        frame_annotated = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # 6. Redimensionar el frame anotado
    resized = cv2.resize(frame_annotated, (orig_w, CAMPO_H))
    if len(warped.shape) == 2: warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    if len(tactical.shape) == 2: tactical = cv2.cvtColor(tactical, cv2.COLOR_GRAY2BGR)

    return np.hstack([resized, warped, tactical])

generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO)

with sv.VideoSink(target_path=OUTPUT_VIDEO, video_info=video_info) as sink:
    for index, frame in enumerate(generator):
        processed_frame = callback_nb12(frame, index)
        sink.write_frame(frame=processed_frame)

print(f"✅ Video guardado sin errores en: {OUTPUT_VIDEO}")



cap = cv2.VideoCapture(OUTPUT_VIDEO)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames_vis = {}
for idx in (0, n_frames // 2, n_frames - 1):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        frames_vis[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cap.release()

_sample = next(iter(frames_vis.values()))
_h, _w = _sample.shape[:2]
_dh = 4
fig, axes = plt.subplots(len(frames_vis), 1,
                         figsize=(_dh * _w / _h, _dh * len(frames_vis)))
titles = ["Frame inicial", "Frame central", "Frame final"]
for ax, (idx, frgb), title in zip(axes, frames_vis.items(), titles):
    ax.imshow(frgb)
    ax.set_title(f"{title} — orig | cenital | máscaras")
    ax.axis("off")
plt.tight_layout()
plt.show()



### Experimento 1: Centroide de máscara vs. centro de bounding box

# Comparar las dos posiciones para cada detección
print("Robot  |  Centro bbox (NB11)  |  Centroide máscara (NB12)  |  Diferencia")
print("─" * 72)
for i, (box, m) in enumerate(zip(dets_sam.xyxy, dets_sam.mask)):
    # NB11: bottom-center del bbox
    cx_box = (box[0] + box[2]) / 2; cy_box = box[3]
    proj_box = cv2.perspectiveTransform(np.float32([[[cx_box, cy_box]]]), H)
    pos_box  = (int(proj_box[0][0][0]), int(proj_box[0][0][1]))

    # NB12: centroide de la máscara
    ys, xs  = np.where(m)
    cx_m, cy_m = float(xs.mean()), float(ys.mean())
    proj_m  = cv2.perspectiveTransform(np.float32([[[cx_m, cy_m]]]), H)
    pos_m   = (int(proj_m[0][0][0]), int(proj_m[0][0][1]))

    diff = np.linalg.norm(np.array(pos_box) - np.array(pos_m))
    cid  = int(dets_sam.class_id[i]) if dets_sam.class_id is not None else 0
    print(f"  {CLASS_NAMES.get(cid,'?'):5s} | {str(pos_box):20s} | {str(pos_m):26s} | {diff:.1f} px")

# 💭 Reflexión: ¿en qué robots hay más diferencia entre los dos métodos?
# ¿El centroide de la máscara es siempre más preciso que el bottom-center?
# ¿Para qué tipo de análisis importa más esta diferencia?



### Experimento 2: Área de cada robot en el campo canónico

print("Robot  | Área cámara | Área canónica | Ratio")
print("─" * 52)
for i, (m, cid) in enumerate(zip(dets_sam.mask, dets_sam.class_id.all() or [])):
    m_uint8  = m.astype(np.uint8) * 255
    m_warped = cv2.warpPerspective(m_uint8, H, (CAMPO_W, CAMPO_H))
    area_cam   = int(m.sum())
    area_canon = int((m_warped > 0).sum())
    ratio = area_canon / area_cam if area_cam > 0 else 0
    print(f"  {CLASS_NAMES.get(int(cid),'?'):5s}  | {area_cam:11d} | {area_canon:13d} | {ratio:.3f}")

total_campo = CAMPO_W * CAMPO_H
print(f"\nÁrea total del campo canónico: {total_campo} px ({CAMPO_W}×{CAMPO_H})")

# 💭 Reflexión: ¿los robots más alejados de la cámara tienen menor área canónica?
# ¿Por qué el ratio difiere entre robots en distintas posiciones de la cancha?
# (Pista: la distorsión de perspectiva comprime las áreas lejanas al proyectarlas)



### Experimento 3: ¿Cuánto campo controla cada equipo?

def team_control(dets_sam: sv.Detections, H: np.ndarray,
                 campo_w: int = CAMPO_W, campo_h: int = CAMPO_H) -> dict:
    """
    Calcula el % del campo canónico cubierto por cada equipo.
    Devuelve dict {class_id: porcentaje}.
    """
    masks     = dets_sam.mask
    class_ids = (dets_sam.class_id if dets_sam.class_id is not None
                 else np.zeros(len(dets_sam), dtype=int))
    total     = campo_w * campo_h
    areas     = {0: 0, 1: 0, 2: 0}

    if masks is not None:
        for m, cid in zip(masks, class_ids):
            m_warped = cv2.warpPerspective(m.astype(np.uint8) * 255, H,
                                           (campo_w, campo_h))
            areas[int(cid)] += int((m_warped > 0).sum())

    return {cid: round(area / total * 100, 3) for cid, area in areas.items()}

control = team_control(dets_sam, H)
print("Control del campo canónico por equipo/objeto:")
for cid, pct in control.items():
    bar = "█" * int(pct * 5)
    print(f"  {CLASS_NAMES.get(cid,'?'):5s}: {pct:6.3f}%  {bar}")

# 💭 Reflexión: esta métrica es el "% de zona controlada" que usan
# los analistas de fútbol profesional. ¿Qué limitaciones tiene
# calcularlo solo con la posición instantánea de un frame?



# RETO RESUELTO: mapa de calor de control por equipo

def build_heatmap(video_path: str, sam_model, H: np.ndarray,
                  campo_w: int = CAMPO_W, campo_h: int = CAMPO_H,
                  max_frames: int = 24) -> dict:
    """
    Procesa los primeros max_frames del video y acumula la presencia
    de cada equipo en el campo canónico.
    Devuelve {class_id: ndarray (campo_h, campo_w) acumulado}.
    """
    heatmaps = {0: np.zeros((campo_h, campo_w), dtype=np.float32),
                1: np.zeros((campo_h, campo_w), dtype=np.float32)}
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        dets = detect_robots_hsv(frame)
        if len(dets) > 0:
            results  = sam_model(frame, bboxes=dets.xyxy.tolist(), verbose=False)
            dets_sam = sv.Detections.from_ultralytics(results[0])
            if len(dets_sam) == len(dets):
                dets_sam.class_id = dets.class_id
            if dets_sam.mask is not None:
                for m, cid in zip(dets_sam.mask, dets_sam.class_id.all() or []):
                    if int(cid) in heatmaps:
                        m_w = cv2.warpPerspective(m.astype(np.uint8) * 255, H,
                                                  (campo_w, campo_h))
                        heatmaps[int(cid)] += m_w.astype(np.float32)
        frame_count += 1
    cap.release()
    # Normalizar a [0, 1]
    for cid in heatmaps:
        mx = heatmaps[cid].max()
        if mx > 0:
            heatmaps[cid] /= mx
    return heatmaps

print("Generando mapas de calor (primeros 24 frames)...")
heatmaps = build_heatmap("assets/futbot-2s.mp4", sam_model, H)

fig, axes = plt.subplots(1, 2, figsize=(8, 6))
for ax, (cid, hm) in zip(axes, heatmaps.items()):
    warped_rgb = cv2.cvtColor(
        cv2.warpPerspective(image_bgr, H, (CAMPO_W, CAMPO_H)),
        cv2.COLOR_BGR2RGB)
    ax.imshow(warped_rgb, alpha=0.4)
    cmap = "Blues" if cid == 0 else "Reds"
    ax.imshow(hm, alpha=0.6, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(f"Equipo {CLASS_NAMES[cid]} — mapa de calor (24 frames)")
    ax.axis("off")
plt.suptitle("Control del campo a lo largo del video", fontsize=13)
plt.tight_layout()
plt.show()
