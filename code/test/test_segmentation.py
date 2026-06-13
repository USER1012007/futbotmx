import cv2
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from infra.configs import Config
from vision.segmentation import SegmentationEngine
from domain.entities import FrameResult

def test_segmentation():
    cfg = Config()
    
    try:
        engine = SegmentationEngine(cfg)
    except Exception as e:
        print(f"Error inicializando SegmentationEngine: {e}")
        return
    
    image_path = "data/videos/futbot-01.jpg"
    if not os.path.exists(image_path):
        print(f"Imagen de prueba no encontrada en {image_path}. Prueba abortada.")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: cv2.imread no pudo cargar la imagen en {image_path}")
        return

    print(f"Imagen cargada correctamente: {frame.shape}")

    try:
        result = engine.process_frame(frame, frame_id=0)
    except Exception:
        import traceback
        traceback.print_exc()
        return

    assert isinstance(result, FrameResult)
    print(f"FrameResult generado con éxito.")
    print(f"Robots detectados: {len(result.robots)}")
    if result.ball:
        print(f"Pelota detectada en: {result.ball.position}")
    else:
        print("No se detectó pelota.")


if __name__ == "__main__":
    test_segmentation()
