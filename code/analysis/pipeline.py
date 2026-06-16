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
        
        # Suscribirse a los resultados del proceso de visión
        self.event_bus.subscribe("frame_processed", self.process)

    def process(self, frame_result: FrameResult):
        # 1. Enriquecer datos
        self.enricher.enrich(frame_result)
        
        # 2. Detectar eventos
        events = self.event_detector.detect(frame_result)
        
        # 3. Publicar eventos detectados al bus
        if events:
            self.event_bus.publish("frame_events", events)
            
        # 4. Actualizar estadísticas
        self.stats.update(frame_result, events)
