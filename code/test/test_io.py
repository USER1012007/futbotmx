from infra.configs import Config
from io_utils.tracking_io import TrackingIO
from domain.entities import Robot
import shutil

def test_io_integration():
    cfg = Config()
    
    test_file = cfg.TRACKING_DIR / "test_tracking.jsonl"

    if test_file.exists():
        test_file.unlink()
        
    io_handler = TrackingIO(test_file)
    
    for i in range(10):
        robot = Robot(id=f"robot_{i}", position=(i, i*2))
        io_handler.save_frame_data(frame_id=i, data={"robot_id": robot.id, "pos": robot.position})
    
    results = io_handler.read_all()
    #
    # assert len(results) == 1
    # assert results[0]["frame_id"] == 1
    # assert results[0]["data"]["robot_id"] == "robot_1"
    #
    print(results)
    
    print("TrackingIO test passed")
    

if __name__ == "__main__":
    test_io_integration()
