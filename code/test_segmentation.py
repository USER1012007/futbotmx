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
    
    # 2. Cargar imagen de prueba (usamos un asset existente si hay)
    # Por ahora, asegurémonos de que exista un frame
    image_path = "./data/videos/futbot-01.jpg"
    if not os.path.exists(image_path):
        print(f"Imagen de prueba no encontrada en {image_path}. Prueba fallida.")
        return

    frame = cv2.imread(image_path)
    
    # 3. Ejecutar procesamiento
    print("Procesando frame con YOLO/SAM...")
    result = engine.process_frame(frame, frame_id=0)
    
    # 4. Validar
    assert isinstance(result, FrameResult)
    print(f"FrameResult generado con éxito:")
    print(f"   - Robots detectados: {len(result.robots)}")
    print(f"   - Pelota detectada: {result.ball is not None}")

if __name__ == "__main__":
    test_segmentation()
