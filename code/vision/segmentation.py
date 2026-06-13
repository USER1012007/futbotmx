import cv2
import supervision as sv
import torch
import numpy as np
from ultralytics import YOLO, SAM
from typing import Optional
from domain.entities import FrameResult, Robot, Ball
from infra.configs import Config
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class SegmentationEngine:
    def __init__(self, cfg: Config):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_model = YOLO("yolov8x.pt") 
        self.sam_model = SAM(cfg.SAM_MODEL_NAME)
        self.cfg = cfg

    def process_frame(self, frame: cv2.typing.MatLike, frame_id: int) -> FrameResult:
        # 1. Detección con YOLO
        yolo_results = self.yolo_model(frame, device=self.device, verbose=True, conf=0.1)[0]
        print(f"DEBUG: YOLO detectó {len(yolo_results.boxes)} cajas con conf=0.1.")
        for i, box in enumerate(yolo_results.boxes):
            print(f"DEBUG: Box {i}: class={int(box.cls[0])}, conf={float(box.conf[0]):.2f}")
            
        detections = sv.Detections.from_ultralytics(yolo_results)
        
        # Filtrar por confianza
        mask = detections.confidence >= self.cfg.DETECTION_THRESHOLD
        detections = detections[mask]
        
        # 2. Conversión a estructuras del dominio
        robots = []
        ball = None
        
        for i in range(len(detections)):
            class_id = int(detections.class_id[i])
            print(f"DEBUG: Detectada clase {class_id} con confianza {detections.confidence[i]:.2f}")
            xyxy = detections.xyxy[i]
            # Centroide del box
            center_x = (xyxy[0] + xyxy[2]) / 2
            center_y = (xyxy[1] + xyxy[3]) / 2
            
            if class_id == 0:  # Persona (Robot)
                robots.append(Robot(id=f"robot_{i}", team_id="unknown", position=(float(center_x), float(center_y))))
            elif class_id == 32: # sports ball
                ball = Ball(id="ball", position=(float(center_x), float(center_y)))
                
        return FrameResult(frame_id=frame_id, robots=robots, ball=ball)

    def detect_robots_hsv(self, frame_bgr: np.ndarray, min_area: int = 40) -> sv.Detections:
        if frame_bgr is None:
            raise ValueError("frame_bgr is None!")

        h_img, w_img = frame_bgr.shape[:2]
        frame_blur = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        # ── Campo verde/cyan (lo que NO son objetos) ──────────────────────────
        mask_campo = cv2.inRange(hsv, np.array([75, 120, 80]), np.array([100, 255, 220]))

        # ── Balón: naranja-rojo puro, pequeño ─────────────────────────────────
        mask_balon  = cv2.inRange(hsv, np.array([  0, 150,  80]), np.array([ 15, 255, 255]))
        mask_balon |= cv2.inRange(hsv, np.array([165, 150,  80]), np.array([179, 255, 255]))

        # ── Portería amarilla: H=20-36, saturada ──────────────────────────────
        mask_port_am = cv2.inRange(hsv, np.array([18, 100,  80]), np.array([38, 255, 255]))

        # ── Portería azul-verde oscura (abajo): H=100-115, S=120-220, V<160 ──
        mask_port_az = cv2.inRange(hsv, np.array([98, 100, 40]), np.array([118, 230, 170]))

        # ── Robots: objetos no-campo dentro del área de juego ─────────────────
        # Invertir campo + excluir marcas blancas (V muy alto) + excluir objetos ya detectados
        mask_no_campo = cv2.bitwise_not(mask_campo)
        mask_blanco   = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([179, 60, 255]))
        mask_negro    = cv2.inRange(hsv, np.array([0, 0,   0]), np.array([179, 255, 40]))
        mask_robots_raw = cv2.bitwise_and(mask_no_campo,
                          cv2.bitwise_not(cv2.bitwise_or(mask_blanco, mask_negro)))
        # Excluir también balón y porterías del mask de robots
        mask_robots_raw = cv2.bitwise_and(mask_robots_raw,
                          cv2.bitwise_not(cv2.bitwise_or(mask_balon, mask_port_am)))
        mask_robots_raw = cv2.bitwise_and(mask_robots_raw,
                          cv2.bitwise_not(mask_port_az))

        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        xyxy_list, class_ids = [], []

        # robots: class_id=0, balón: class_id=1, portería amarilla: class_id=2, portería azul: class_id=3
        configs = [
            # (mask, kernel_close, kernel_open, cid, min_area, max_area_frac)
            (mask_robots_raw, kernel9, kernel5, 0, 800,  0.04),
            (mask_balon,      kernel5, kernel5, 1,  30,  0.005),
            (mask_port_am,    kernel9, kernel5, 2, 600,  0.25),
            (mask_port_az,    kernel9, kernel5, 3, 400,  0.20),
        ]

        for mask, kern_c, kern_o, cid, min_a, max_frac in configs:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern_c)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kern_o)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = w_img * h_img * max_frac
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_a or area > max_area:
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

    def segment_with_sam(self, image_bgr: np.ndarray, dets_hsv: sv.Detections) -> Optional[sv.Detections]:

        CLASS_NAMES = {0: "azul", 1: "rojo", 2: "balón", 3: "portería"}
        COLORS_HEX  = {0: "#00b4d8", 1: "#ef233c", 2: "#ff9500", 3: "#ffd000"}
        COLORS_BGR  = {0: (216, 180, 0), 1: (60, 35, 239), 2: (0, 149, 255), 3: (0, 208, 255)}

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sam_model = SAM("sam3.pt")

        bboxes = dets_hsv.xyxy.tolist()
        results = sam_model(image_bgr, bboxes=bboxes, verbose=False)
        dets_sam = sv.Detections.from_ultralytics(results[0])

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

