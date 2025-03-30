from dataclasses import dataclass

from src.wellknown_diagram import WellKnownDiagram


@dataclass(frozen=True, slots=True)
class Outcome:
    diagram_id: str
    body: str
