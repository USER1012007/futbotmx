from typing import Optional
from domain.entities import FrameResult
from analysis.enricher import DataEnricher
from analysis.stats_engine import StatsEngine
from analysis.event_detector import EventDetector
from analysis.referee_engine import RefereeEngine
from infra.event_bus import EventBus

class AnalysisPipeline:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.enricher = DataEnricher()
        self.stats = StatsEngine(event_bus)
        self.event_detector = EventDetector()
        self.referee = RefereeEngine(event_bus)
        
        self.event_bus.subscribe("frame_processed", self.process)

    def process(self, frame_result: FrameResult):
        self.enricher.enrich(frame_result)
        
        events = self.event_detector.detect(frame_result)
        if events:
            self.event_bus.publish("frame_events", events)
            
        self.stats.update(frame_result, events)
