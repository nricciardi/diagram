from typing import List

from src.wellknown_diagram import WellKnownDiagram
from core.representation.representation import DiagramRepresentation
from src.representation.flowchart_representation.flowchart_representation import FlowchartRepresentation
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
        assert isinstance(diagram_representation, FlowchartRepresentation)
        
        body: str = ""
        
        #TODO: Stuff to do with the diagram representation
        
        outcome: Outcome = Outcome(diagram_id, body)
        return outcome

