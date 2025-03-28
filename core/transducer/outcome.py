from dataclasses import dataclass

from src.wellknown_diagram import WellKnownDiagram


@dataclass
class Outcome:
    diagram_id: str
    body: str
