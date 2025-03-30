from typing import List
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.wellknown_diagram import WellKnownDiagram
from core.representation.representation import DiagramRepresentation
from src.representation.flowchart_representation.flowchart_representation import FlowchartRepresentation
from src.unified_representation import UnifiedDiagramRepresentation
from core.transducer.outcome import Outcome
from core.transducer.transducer import Transducer

from src.representation.flowchart_representation.element import FlowchartElement, FlowchartRelation

class FlowchartToMermaidTransducer(Transducer):

    def __init__(self, identifier: str):
        super().__init__(identifier)

    def compatible_diagrams(self) -> List[str]:
        return [
            WellKnownDiagram.FLOW_CHART.value,
            WellKnownDiagram.GRAPH_DIAGRAM.value,
        ]

    @staticmethod
    def wrap_element(category: FlowchartElement, label: str) -> str:
        match category:
            case FlowchartElement.CIRCLE:
                return f"(({label}))"
            case FlowchartElement.TERMINAL:
                return f"([{label}])"
            case FlowchartElement.PROCESS:
                return f"({label})"
            case FlowchartElement.DECISION:
                return "{" + label + "}"
            case FlowchartElement.INPUT_OUTPUT:
                return f"[/{label}/]"
            case FlowchartElement.SUBROUTINE:
                return f"[{label}]"
            case _:
                raise ValueError(f"Unknown flowchart element category: {category}")

    @staticmethod
    def wrap_relation(category: FlowchartRelation, label: str) -> str:
        match category:
            case FlowchartRelation.ARROW:
                if label == "":
                    return "-->"
                return f"-->|{label}|"
            case FlowchartRelation.OPEN_LINK:
                if label == "":
                    return " --- "
                return f"---|{label}|"
            case FlowchartRelation.DOTTED_ARROW:
                if label == "":
                    return "-.->"
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

import unittest
from src.representation.flowchart_representation.flowchart_representation import FlowchartElement, FlowchartRelation, FlowchartElementId
from typing import Dict

class TestFlowchartToMermaidTransducer(unittest.TestCase):
    
    def setUp(self):
        self.transducer = FlowchartToMermaidTransducer("test_transducer")
    
    def test_compatible_diagrams(self):
        compatible_diagrams = self.transducer.compatible_diagrams()
        assert WellKnownDiagram.FLOW_CHART.value in compatible_diagrams, "Flowchart should be compatible"
        assert WellKnownDiagram.GRAPH_DIAGRAM.value in compatible_diagrams, "Graph diagram should be compatible"

    def test_wrap_element(self):
        assert self.transducer.wrap_element(FlowchartElement.CIRCLE, "Test") == "((Test))", \
            "Wrapped element should be ((Test))"
        assert self.transducer.wrap_element(FlowchartElement.TERMINAL, "Test") == "([Test])", \
            "Wrapped element should be ([Test])"

    def test_wrap_relation(self):
        wrapped_relation = self.transducer.wrap_relation(FlowchartRelation.ARROW, "Test")
        assert wrapped_relation == "-->|Test|", "Wrapped relation should be -->|Test|"
        
    def test_elaborate(self):
        
        id_a, id_b, id_c = FlowchartElementId("A"), FlowchartElementId("B"), FlowchartElementId("C")
        
        elements: Dict[str, FlowchartElement] = {
            id_a: FlowchartElement(id_a, FlowchartElement.CIRCLE, "Start_Node"),
            id_b: FlowchartElement(id_b, FlowchartElement.PROCESS, "i++"),
            id_c: FlowchartElement(id_c, FlowchartElement.DECISION, "if i > 5"),
        }
        
        relations: List[FlowchartRelation] = [
            FlowchartRelation(FlowchartRelation.ARROW, id_a, id_b, "int i = 0"),
            FlowchartRelation(FlowchartRelation.DOTTED_ARROW, id_b, id_c, ""),
        ]
        
        representation = FlowchartRepresentation(elements, relations)
        expected_outcome = Outcome("test_diagram", "Flowchart TD\n" \
                          "\tA((Start_Node))\n" \
                          "\tB(i++)\n" \
                          "\tC{if i > 5}\n" \
                          "\n" \
                          "\tA-->|int i = 0|B\n" \
                          "\tB-.->C\n")
        assert expected_outcome == self.transducer.elaborate("test_diagram", representation), \
            "Elaborated outcome should match expected outcome"
            
if __name__ == "__main__":
    unittest.main()