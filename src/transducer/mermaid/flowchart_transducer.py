from typing import List

from src.diagram import Diagram
from src.representation.representation import DiagramRepresentation
from src.transducer.outcome import Outcome
from src.transducer.transducer import Transducer


class FlowchartToMermaidTransducer(Transducer):

    def compatible_diagrams(self) -> List[str]:
        return [
            Diagram.FLOW_CHART.value,
            Diagram.GRAPH_DIAGRAM.value,
        ]

    def elaborate(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> Outcome:
        pass        # TODO: il codice per ottenere il mermaid del flow chart (e graph) qui

