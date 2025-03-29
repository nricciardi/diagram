from typing import List

from src.wellknown_diagram import WellKnownDiagram
from core.representation.representation import DiagramRepresentation
from src.representation.flowchart_representation.flowchart_representation import FlowchartRepresentation
from src.unified_representation import UnifiedDiagramRepresentation
from core.transducer.outcome import Outcome
from core.transducer.transducer import Transducer

from src.representation.flowchart_representation.flowchart_enum import FlowchartElementEnum, FlowchartRelationEnum

class FlowchartToMermaidTransducer(Transducer):

    def __init__(self, identifier: str):
        super().__init__(identifier)

    def compatible_diagrams(self) -> List[str]:
        return [
            WellKnownDiagram.FLOW_CHART.value,
            WellKnownDiagram.GRAPH_DIAGRAM.value,
        ]

    @staticmethod
    def wrap_element(category: FlowchartElementEnum, label: str) -> str:
        match category:
            case FlowchartElementEnum.CIRCLE:
                return f"(({label}))"
            case FlowchartElementEnum.TERMINAL:
                return f"([{label}])"
            case FlowchartElementEnum.PROCESS:
                return f"({label})"
            case FlowchartElementEnum.DECISION:
                return "{" + label + "}"
            case FlowchartElementEnum.INPUT_OUTPUT:
                return f"[/{label}/]"
            case FlowchartElementEnum.SUBROUTINE:
                return f"[{label}]"
            case _:
                raise ValueError(f"Unknown flowchart element category: {category}")

    @staticmethod
    def wrap_relation(category: FlowchartRelationEnum, label: str) -> str:
        match category:
            case FlowchartRelationEnum.ARROW:
                return f"-->|{label}|"
            case FlowchartRelationEnum.OPEN_LINK:
                return f"---|{label}|"
            case FlowchartRelationEnum.DOTTED_ARROW:
                return f" -. {label} .->"
            case _:
                raise ValueError(f"Unknown flowchart relation category: {category}")

    def elaborate(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> Outcome:
        assert isinstance(diagram_representation, FlowchartRepresentation)
        
        body: str = "Flowchart TD\n"
        for id, element in diagram_representation.elements.items():
            body += f"\t{id.id}{self.wrap_element(element.category, element.label)}\n"
            
        body += "\n"
        for relation in diagram_representation.relations:
            body += f"\t{relation.source_id.id}"
            body += f"{self.wrap_relation(relation.category, relation.label)}{relation.target_id.id}\n"
        
        outcome: Outcome = Outcome(diagram_id, body)
        return outcome

