from typing import List, Type
from py_d2 import D2Diagram, D2Shape, D2Connection, D2Style

from src.wellknown_diagram import WellKnownDiagram
from core.representation.representation import DiagramRepresentation
from src.representation.flowchart_representation.flowchart_representation import FlowchartRepresentation
from core.transducer.outcome import TransducerOutcome
from core.transducer.transducer import Transducer
from src.wellknown_markuplang import WellKnownMarkupLanguage

class FlowchartToD2Transducer(Transducer):

    def __init__(self, identifier: str):
        super().__init__(identifier)

    def compatible_diagrams(self) -> List[str]:
        return [
            WellKnownDiagram.FLOW_CHART.value,
            WellKnownDiagram.GRAPH_DIAGRAM.value,
        ]

    def compatible_representations(self) -> List[Type[DiagramRepresentation]]:
        return [FlowchartRepresentation]

    def transduce(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> TransducerOutcome:
        assert isinstance(diagram_representation, FlowchartRepresentation)

        shapes: list[D2Shape] = [
            D2Shape(
                name=element.identifier,
                label=element.label,

            ) for identifier, element in diagram_representation.elements.items()
        ]

        connections: list[D2Connection] = [
            D2Connection(
                shape_1=relation.source_id,
                shape_2=relation.target_id,
                label=relation.label
            ) for relation in diagram_representation.relations
        ]

        body: D2Diagram = D2Diagram(shapes=shapes, connections=connections)

        outcome: TransducerOutcome = TransducerOutcome(diagram_id, WellKnownMarkupLanguage.D2_LANG.value, payload=str(body))
        return outcome