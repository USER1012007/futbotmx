import cv2
from infra.configs import Config
from vision.segmentation import SegmentationEngine
from domain.entities import FrameResult
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_segmentation():
    # 1. Configuración y motor
    cfg = Config()
    engine = SegmentationEngine(cfg)
    
    # 2. Cargar imagen de prueba
    image_path = "../notebooks/rojas/assets/futbot-01.jpg"
    if not os.path.exists(image_path):
        print(f"Imagen de prueba no encontrada en {image_path}. Prueba fallida.")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: cv2.imread no pudo cargar la imagen en {image_path}")
        return

    print(f"Imagen cargada correctamente: {frame.shape}")

    print(type(frame))
    dets_hsv = engine.detect_robots_hsv(frame)
    print(f"Detecciones HSV: {len(dets_hsv)}  — bboxes que usaremos como prompts para SAM")

    result = engine.segment_with_sam(frame,dets_hsv)

    print(f"FrameResult generado con éxito.")


if __name__ == "__main__":
    test_segmentation()
