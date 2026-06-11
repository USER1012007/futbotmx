import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import supervision as sv
from pathlib import Path

# Cargar imagen de referencia
image_bgr = cv2.imread("assets/futbot-01.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Recalcular H (mismos puntos que NB10 — notebook standalone)
SOURCE_POINTS = np.float32([[20, 165], [590, 40], [592, 1052], [15, 985]])
CAMPO_W, CAMPO_H = 364, 486
ESCALA_PX_CM = 2.0  # 1 cm real = 2 px en el campo canónico (RCJ SF23: 182 × 243 cm)
TARGET_POINTS = np.float32([[0, 0], [CAMPO_W, 0], [CAMPO_W, CAMPO_H], [0, CAMPO_H]])
H = cv2.getPerspectiveTransform(SOURCE_POINTS, TARGET_POINTS)

CLASS_NAMES = {0: "azul", 1: "rojo", 2: "balón"}
COLORS_HEX  = {0: "#00b4d8", 1: "#ef233c", 2: "#ff9500"}
COLORS_BGR  = {0: (216, 180, 0), 1: (60, 35, 239), 2: (0, 149, 255)}

print(f"Imagen: {image_bgr.shape[1]}×{image_bgr.shape[0]} px | H: {H.shape}")
plt.figure(figsize=(5.6, 10))
plt.imshow(image_rgb)
plt.title("futbot-01.jpg — imagen de referencia")
plt.axis("off")
plt.tight_layout()
plt.show()



def detect_robots_hsv(frame_bgr: np.ndarray,
                      min_area: int = 500) -> sv.Detections:
    """
    Detecta robots y balón en un frame usando filtros de color HSV.
    Devuelve sv.Detections con class_id: 0=azul, 1=rojo, 2=balón.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask_azul = cv2.inRange(hsv, np.array([100,  80,  60]),
                                 np.array([125, 255, 255]))
    mask_rojo = cv2.inRange(hsv, np.array([  0, 100,  60]),
                                 np.array([ 10, 255, 255]))
    mask_rojo |= cv2.inRange(hsv, np.array([170, 100,  60]),
                                  np.array([179, 255, 255]))
    mask_balon = cv2.inRange(hsv, np.array([  8, 150, 150]),
                                  np.array([ 25, 255, 255]))

    xyxy_list, class_ids = [], []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for mask, cid in [(mask_azul, 0), (mask_rojo, 1), (mask_balon, 2)]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, w, h_box = cv2.boundingRect(cnt)
            xyxy_list.append([x, y, x + w, y + h_box])
            class_ids.append(cid)

    if not xyxy_list:
        return sv.Detections.empty()

    return sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )

# Probar en la imagen estática
dets = detect_robots_hsv(image_bgr)
print(f"Detecciones encontradas: {len(dets)}")
for i, (box, cid) in enumerate(zip(dets.xyxy, dets.class_id)):
    cx = int((box[0] + box[2]) / 2)
    cy = int(box[3])
    print(f"  [{i}] {CLASS_NAMES[cid]:5s}: bbox={box.astype(int).tolist()}  bottom_center=({cx},{cy})")



# Visualizar detecciones sobre la imagen
vis = image_rgb.copy()
for box, cid in zip(dets.xyxy, dets.class_id):
    x1, y1, x2, y2 = box.astype(int)
    color_bgr = COLORS_BGR[cid]
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    cv2.rectangle(vis, (x1, y1), (x2, y2), color_rgb, 3)
    cx, cy = (x1 + x2) // 2, y2
    cv2.circle(vis, (cx, cy), 8, color_rgb, -1)  # BOTTOM_CENTER
    cv2.putText(vis, CLASS_NAMES[cid], (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_rgb, 2)

plt.figure(figsize=(5.6, 10))
plt.imshow(vis)
plt.title(f"detect_robots_hsv: {len(dets)} detecciones (● = BOTTOM_CENTER)")
plt.axis("off")
plt.tight_layout()
plt.show()



def project_detections(detections: sv.Detections, H: np.ndarray) -> list:
    """
    Proyecta el BOTTOM_CENTER de cada bbox al campo canónico.
    Para el balón (class_id=2) usa el centro geométrico.
    """
    puntos = []
    for box, cid in zip(detections.xyxy, detections.class_id):
        if int(cid) == 2:          # balón: centro geométrico
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
        else:                      # robots: bottom center
            cx = (box[0] + box[2]) / 2
            cy = box[3]
        pt = np.float32([[cx, cy]])
        proj = cv2.perspectiveTransform(pt.reshape(1, 1, 2), H)
        puntos.append({
            "pos": (int(proj[0][0][0]), int(proj[0][0][1])),
            "class_id": int(cid),
        })
    return puntos

puntos_canon = project_detections(dets, H)
print("Proyecciones al campo canónico:")
for p in puntos_canon:
    print(f"  {CLASS_NAMES[p['class_id']]:5s}: {p['pos']}")



def draw_tactical_auto(puntos_canon: list,
                       campo_w: int = CAMPO_W, campo_h: int = CAMPO_H,
                       title: str = "Mapa táctico — detección automática HSV") -> None:
    """Dibuja el campo canónico con posiciones detectadas automáticamente."""
    fig, ax = plt.subplots(figsize=(7, 9.3))
    ax.set_facecolor("#1b4332")
    ax.set_xlim(0, campo_w); ax.set_ylim(campo_h, 0)

    ax.add_patch(mpatches.Rectangle((0, 0), campo_w, campo_h,
                 lw=2, edgecolor="#74c69d", facecolor="none"))
    ax.axhline(y=campo_h / 2, color="#74c69d", lw=1.5)
    ax.add_patch(plt.Circle((campo_w / 2, campo_h / 2), int(30 * ESCALA_PX_CM),
                 color="#74c69d", fill=False, lw=1.5))
    pen_w, pen_h = int(80 * ESCALA_PX_CM), int(40 * ESCALA_PX_CM)
    pen_x = (campo_w - pen_w) / 2
    for y_pen in (0, campo_h - pen_h):
        ax.add_patch(mpatches.Rectangle((pen_x, y_pen), pen_w, pen_h,
                     lw=1, edgecolor="#74c69d", facecolor="none"))
    goal_w = int(60 * ESCALA_PX_CM); goal_x = (campo_w - goal_w) / 2
    ax.add_patch(mpatches.Rectangle((goal_x, 0), goal_w, int(8 * ESCALA_PX_CM),
                 lw=2, edgecolor="#ffd60a", facecolor="#ffd60a33"))
    ax.add_patch(mpatches.Rectangle((goal_x, campo_h - int(8 * ESCALA_PX_CM)), goal_w, int(8 * ESCALA_PX_CM),
                 lw=2, edgecolor="#00b4d8", facecolor="#00b4d833"))

    for p in puntos_canon:
        x, y = p["pos"]
        cid  = p["class_id"]
        color = COLORS_HEX[cid]
        size  = 120 if cid == 2 else 350
        ax.scatter(x, y, s=size, c=color, zorder=5,
                   edgecolors="white", linewidths=1.5)
        if cid != 2:
            lbl = "A" if cid == 0 else "R"
            ax.text(x, y, lbl, ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white", zorder=6)
        else:
            ax.text(x + 14, y - 14, "⚽", fontsize=11, zorder=6)

    ax.set_title(title, fontsize=13, color="white", pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.tick_params(colors="#555")
    plt.tight_layout()
    plt.show()

draw_tactical_auto(puntos_canon)



def draw_tactical_canvas(puntos_canon: list,
                         campo_w: int = CAMPO_W,
                         campo_h: int = CAMPO_H) -> np.ndarray:
    """Dibuja el mapa táctico como imagen BGR (para incrustar en video)."""
    canvas = np.zeros((campo_h, campo_w, 3), dtype=np.uint8)
    canvas[:] = (50, 67, 27)  # verde oscuro en BGR

    cv2.rectangle(canvas, (0, 0), (campo_w - 1, campo_h - 1), (120, 200, 116), 2)
    cv2.line(canvas, (0, campo_h // 2), (campo_w, campo_h // 2), (120, 200, 116), 1)
    cv2.circle(canvas, (campo_w // 2, campo_h // 2), int(30 * ESCALA_PX_CM), (120, 200, 116), 1)

    pen_w = int(80 * ESCALA_PX_CM); pen_h = int(40 * ESCALA_PX_CM)
    pen_x = (campo_w - pen_w) // 2
    cv2.rectangle(canvas, (pen_x, 0), (pen_x + pen_w, pen_h), (120, 200, 116), 1)
    cv2.rectangle(canvas, (pen_x, campo_h - pen_h),
                           (pen_x + pen_w, campo_h - 1), (120, 200, 116), 1)

    goal_w_px = int(60 * ESCALA_PX_CM); goal_x_px = (campo_w - goal_w_px) // 2
    cv2.rectangle(canvas, (goal_x_px, 0), (goal_x_px + goal_w_px, int(8 * ESCALA_PX_CM)), (0, 214, 255), 2)
    cv2.rectangle(canvas, (goal_x_px, campo_h - int(8 * ESCALA_PX_CM)),
                           (goal_x_px + goal_w_px, campo_h - 1), (216, 180, 0), 2)

    for p in puntos_canon:
        x, y = p["pos"]
        cid  = p["class_id"]
        if not (0 <= x < campo_w and 0 <= y < campo_h):
            continue
        color = COLORS_BGR[cid]
        r = 7 if cid == 2 else 13
        cv2.circle(canvas, (x, y), r, color, -1)
        cv2.circle(canvas, (x, y), r, (255, 255, 255), 1)
        if cid != 2:
            lbl = "A" if cid == 0 else "R"
            cv2.putText(canvas, lbl, (x - 5, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return canvas

# Verificar en imagen estática
tactical_bgr = draw_tactical_canvas(puntos_canon)
plt.figure(figsize=(6, 8.0))
plt.imshow(cv2.cvtColor(tactical_bgr, cv2.COLOR_BGR2RGB))
plt.title("draw_tactical_canvas — versión OpenCV para video")
plt.axis("off")
plt.tight_layout()
plt.show()



OUTPUT_VIDEO = "assets/futbot_nb11.mp4"
OUTPUT_VIDEO = "assets/futbot_nb11.mp4"
SOURCE_VIDEO = "assets/futbot-2s.mp4"

video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO)
video_info.width = 998
video_info.height = 486

def callback_nb11(frame: np.ndarray, _: int) -> np.ndarray:
    # 1. Detectar robots y balón
    dets = detect_robots_hsv(frame)
    
    # 2. Proyectar al campo canónico
    puntos = project_detections(dets, H)
    
    # 3. Vista cenital (warped) -> Altura: CAMPO_H
    warped = cv2.warpPerspective(frame, H, (CAMPO_W, CAMPO_H))
    
    # 4. Mapa táctico -> ¡Asegúrate de que mida CAMPO_H de alto!
    tactical = draw_tactical_canvas(puntos) 
    if tactical.shape[0] != CAMPO_H:
        tactical = cv2.resize(tactical, (tactical.shape[1], CAMPO_H))
        
    # Asegurar que tactical tenga 3 canales si es escala de grises
    if len(tactical.shape) == 2:
        tactical = cv2.cvtColor(tactical, cv2.COLOR_GRAY2BGR)
    if len(warped.shape) == 2:
        warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    # 5. Frame original redimensionado
    # Para evitar que orig_w varíe por redondeos, es mejor fijar un ancho estático
    # o confiar en el cálculo si el video original es 100% constante.
    escala = CAMPO_H / frame.shape[0]
    orig_w = int(frame.shape[1] * escala)
    resized = cv2.resize(frame, (orig_w, CAMPO_H))
    
    # Dibujar bboxes sobre el frame redimensionado
    for box, cid in zip(dets.xyxy, dets.class_id):
        x1, y1, x2, y2 = (box * escala).astype(int)
        cv2.rectangle(resized, (x1, y1), (x2, y2), COLORS_BGR[cid], 2)
        
    # Concatenar horizontalmente
    final_frame = np.hstack([resized, warped, tactical])
    
    return final_frame

generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO)

with sv.VideoSink(target_path=OUTPUT_VIDEO, video_info=video_info) as sink:
    for index, frame in enumerate(generator):
        processed_frame = callback_nb11(frame, index)
        sink.write_frame(frame=processed_frame)

print(f"✅ Video guardado con éxito en: {OUTPUT_VIDEO}")



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
    ax.set_title(f"{title} — orig | cenital | táctico")
    ax.axis("off")
plt.tight_layout()
plt.show()



hsv_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

mask_estricto = cv2.inRange(hsv_img, np.array([108, 120, 80]),
                                     np.array([118, 255, 255]))
mask_amplio   = cv2.inRange(hsv_img, np.array([ 90,  50, 40]),
                                     np.array([135, 255, 255]))

fig, axes = plt.subplots(1, 3, figsize=(12, 7))
axes[0].imshow(image_rgb);        axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(mask_estricto, cmap="gray")
axes[1].set_title(f"Estricto H=[108,118] — {mask_estricto.sum()//255} px activos")
axes[1].axis("off")
axes[2].imshow(mask_amplio, cmap="gray")
axes[2].set_title(f"Amplio H=[90,135] — {mask_amplio.sum()//255} px activos")
axes[2].axis("off")
plt.suptitle("Experimento 1: rango HSV del equipo azul", fontsize=12)
plt.tight_layout(); plt.show()



for min_area in (100, 300, 500, 1500):
    d = detect_robots_hsv(image_bgr, min_area=min_area)
    n_azul = sum(1 for c in d.class_id if c == 0)
    n_rojo = sum(1 for c in d.class_id if c == 1)
    n_balon = sum(1 for c in d.class_id if c == 2)
    print(f"min_area={min_area:5d} → total={len(d):2d}  "
          f"(azul={n_azul}, rojo={n_rojo}, balón={n_balon})")



# Sin tracking: cada frame asigna IDs locales 0, 1, 2, ...
# Un robot que desaparece un frame y reaparece recibe un ID diferente.
# Con ByteTrack: el robot mantiene su ID aunque no se detecte brevemente.

from trackers import ByteTrackTracker

tracker = ByteTrackTracker()
cap = cv2.VideoCapture("assets/futbot-2s.mp4")

ids_por_frame = []
for i in range(15):
    ret, frame = cap.read()
    if not ret:
        break
    d = detect_robots_hsv(frame)
    if len(d) > 0:
        d = tracker.update(d)
        ids = list(d.tracker_id) if d.tracker_id is not None else []
    else:
        ids = []
    ids_por_frame.append((i, ids))
cap.release()

print("tracker_id por frame (primeros 15 frames):")
for i, ids in ids_por_frame:
    print(f"  Frame {i:02d}: {ids}")



def dist_cm(p1: tuple, p2: tuple, escala: float = ESCALA_PX_CM) -> float:
    """Distancia euclidiana entre dos puntos canónicos, en centímetros reales."""
    dx = (p2[0] - p1[0]) / escala
    dy = (p2[1] - p1[1]) / escala
    return float(np.sqrt(dx**2 + dy**2))

# Leer 2 frames consecutivos del video de entrada para estimar velocidad
cap_exp = cv2.VideoCapture("assets/futbot-2s.mp4")
frames_exp = []
for _ in range(2):
    ret, frm = cap_exp.read()
    if ret:
        frames_exp.append(frm)
cap_exp.release()

if len(frames_exp) == 2:
    pts0 = project_detections(detect_robots_hsv(frames_exp[0]), H)
    pts1 = project_detections(detect_robots_hsv(frames_exp[1]), H)
    print(f"Campo: {CAMPO_W/ESCALA_PX_CM:.0f} × {CAMPO_H/ESCALA_PX_CM:.0f} cm real")
    print()
    # Distancias entre objetos en frame 0
    for i, pi in enumerate(pts0):
        for j, pj in enumerate(pts0):
            if j <= i:
                continue
            d = dist_cm(pi["pos"], pj["pos"])
            print(f"  {CLASS_NAMES[pi['class_id']]} → {CLASS_NAMES[pj['class_id']]}: {d:.1f} cm")
    print()
    # Velocidad del primer objeto detectado entre frame 0 y frame 1
    if pts0 and pts1:
        v = dist_cm(pts0[0]["pos"], pts1[0]["pos"]) * 30.0  # 30 fps
        print(f"  {CLASS_NAMES[pts0[0]['class_id']]} vel. entre frames: {v:.1f} cm/s  ({v/100:.2f} m/s)")
else:
    print("No se pudo leer el video — verifica la ruta de assets/futbot-2s.mp4")



# Solución del reto — detector basado en YOLO (misma interfaz que detect_robots_hsv)

from ultralytics import YOLO

def detect_with_yolo(frame_bgr: np.ndarray,
                     model_path: str = "yolov8n.pt",
                     conf: float = 0.3) -> sv.Detections:
    model = YOLO(model_path)
    results = model(frame_bgr, conf=conf, verbose=False)[0]
    return sv.Detections.from_ultralytics(results)

  # dets = detect_with_yolo(frame)

print("detect_with_yolo definida. Intercambio de detectores — mismo pipeline, distinta fuente:")
print("  NB11 (color):", "detect_robots_hsv(frame)  → sv.Detections")
print("  Reto (YOLO): ", "detect_with_yolo(frame)   → sv.Detections")
print("  project_detections() y draw_tactical_canvas() no cambian.")
