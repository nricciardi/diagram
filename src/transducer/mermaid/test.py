import unittest

from core.transducer.outcome import Outcome
from src.representation.flowchart_representation.element import FlowchartElementCategory
from src.representation.flowchart_representation.flowchart_representation import Element, Relation, \
    FlowchartRepresentation
from typing import Dict, List

from src.representation.flowchart_representation.relation import FlowchartRelationCategory
from src.transducer.mermaid.flowchart_transducer import FlowchartToMermaidTransducer
from src.wellknown_diagram import WellKnownDiagram


class TestFlowchartToMermaidTransducer(unittest.TestCase):

    def setUp(self):
        self.transducer = FlowchartToMermaidTransducer("test_transducer")

    def test_compatible_diagrams(self):
        compatible_diagrams = self.transducer.compatible_diagrams()
        assert WellKnownDiagram.FLOW_CHART.value in compatible_diagrams, "Flowchart should be compatible"
        assert WellKnownDiagram.GRAPH_DIAGRAM.value in compatible_diagrams, "Graph diagram should be compatible"

    def test_wrap_element(self):
        assert self.transducer.wrap_element(FlowchartElementCategory.CIRCLE.value, "Test") == "((Test))", \
            "Wrapped element should be ((Test))"
        assert self.transducer.wrap_element(FlowchartElementCategory.TERMINAL.value, "Test") == "([Test])", \
            "Wrapped element should be ([Test])"

    def test_wrap_relation(self):
        wrapped_relation = self.transducer.wrap_relation(FlowchartRelationCategory.ARROW.value, "Test")
        assert wrapped_relation == "-->|Test|", "Wrapped relation should be -->|Test|"

    def test_elaborate(self):
        id_a, id_b, id_c = "A", "B", "C"

        elements: Dict[str, Element] = {
            id_a: Element(id_a, FlowchartElementCategory.CIRCLE.value, "Start_Node"),
            id_b: Element(id_b, FlowchartElementCategory.PROCESS.value, "i++"),
            id_c: Element(id_c, FlowchartElementCategory.DECISION.value, "if i > 5"),
        }

        relations: List[Relation] = [
            Relation(FlowchartRelationCategory.ARROW.value, id_a, id_b, "int i = 0"),
            Relation(FlowchartRelationCategory.DOTTED_ARROW.value, id_b, id_c, ""),
        ]

        representation = FlowchartRepresentation(elements, relations)
        expected_outcome = Outcome("test_diagram", "Flowchart TD\n"
                                                   "\tA((Start_Node))\n"
                                                   "\tB(i++)\n"
                                                   "\tC{if i > 5}\n"
                                                   "\n"
                                                   "\tA-->|int i = 0|B\n"
                                                   "\tB-.->C\n")

        assert expected_outcome == self.transducer.elaborate("test_diagram", representation), \
            "Elaborated outcome should match expected outcome"


if __name__ == "__main__":
    unittest.main()