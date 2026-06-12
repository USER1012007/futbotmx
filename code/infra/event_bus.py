from typing import Callable, Any, Dict, List
from collections import defaultdict

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callable[[Any], None]):
        self._subscribers[event_type].append(callback)

    def publish(self, event_type: str, data: Any):
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                callback(data)
