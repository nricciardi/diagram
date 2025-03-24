from dataclasses import dataclass

from src.diagram import Diagram


@dataclass
class Outcome:
    diagram_id: str
    body: str
