from dataclasses import dataclass

from src.wellknown_diagram import WellKnownDiagram


@dataclass(frozen=True, slots=True)
class TransducerOutcome:
    diagram_id: str
    markup_language: str
    payload: str
