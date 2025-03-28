from typing import List

from src.wellknown_diagram import WellKnownDiagram
from core.representation.representation import DiagramRepresentation
from src.unified_representation import UnifiedDiagramRepresentation
from core.transducer.outcome import Outcome
from core.transducer.transducer import Transducer


class FlowchartToMermaidTransducer(Transducer):

    def __init__(self, identifier: str):
        super().__init__(identifier)

    def compatible_diagrams(self) -> List[str]:
        return [
            WellKnownDiagram.FLOW_CHART.value,
            WellKnownDiagram.GRAPH_DIAGRAM.value,
        ]

    def elaborate(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> Outcome:
        assert isinstance(diagram_representation, UnifiedDiagramRepresentation)
        # TODO: il codice per ottenere il mermaid del flow chart (e graph) qui

