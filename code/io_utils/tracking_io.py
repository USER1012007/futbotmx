import json
from pathlib import Path
from typing import Dict, Any

class TrackingIO:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def save_frame_data(self, frame_id: int, data: Dict[str, Any]):
        entry = {"frame_id": frame_id, "data": data}
        with open(self.file_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def read_all(self):
        results = []
        if not self.file_path.exists():
            return results
            
        with open(self.file_path, "r") as f:
            for line in f:
                results.append(json.loads(line))
        return results
