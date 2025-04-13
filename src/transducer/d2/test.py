import unittest
from typing import Dict, List

from src.wellknown_markuplang import WellKnownMarkupLanguage
from src.transducer.d2.flowchart_transducer import FlowchartToD2Transducer
from src.wellknown_diagram import WellKnownDiagram
from src.representation.flowchart_representation.element import FlowchartElementCategory, Element
from src.representation.flowchart_representation.relation import FlowchartRelationCategory, Relation
from src.representation.flowchart_representation.flowchart_representation import FlowchartRepresentation
from core.transducer.outcome import TransducerOutcome

class TestFlowchartToD2LTransducer(unittest.TestCase):

    def setUp(self):
        self.transducer = FlowchartToD2Transducer("test_transducer")

    def test_compatible_diagrams(self):
        compatible_diagrams = self.transducer.compatible_diagrams()
        assert WellKnownDiagram.FLOW_CHART.value in compatible_diagrams, "Flowchart should be compatible"
        assert WellKnownDiagram.GRAPH_DIAGRAM.value in compatible_diagrams, "Graph diagram should be compatible"

    def test_wrap_element(self):
        id_a, id_f = "A", "F"
        result: str = self.transducer.wrap_element(FlowchartElementCategory.CIRCLE.value, "Test", id_a)
        expected_result: str = "A: Test\nA.shape: circle\n"
        self.assertEqual(result, expected_result, f"Expected\n{expected_result}, got\n{result}")
        result: str = self.transducer.wrap_element(FlowchartElementCategory.TERMINAL.value, "Test", id_f)
        expected_result: str = "F: Test\nF.shape: oval\n"
        self.assertEqual(result, expected_result)

    def test_wrap_relation(self):
        id_d = "D"
        result = self.transducer.wrap_relation(FlowchartRelationCategory.ARROW.value, "Test", id_d)
        expected_result: str = "->D: Test\n"
        self.assertEqual(result, expected_result)

    def test_transduce(self):
        id_a, id_b, id_c, id_d, id_e, id_f = "0", "1", "2", "3", "4", "5"

        elements: list[Element] = [
            Element(FlowchartElementCategory.CIRCLE.value, ["Circle"], []),
            Element(FlowchartElementCategory.PROCESS.value, ["Process"], []),
            Element(FlowchartElementCategory.DECISION.value, ["Decision"], []),
            Element(FlowchartElementCategory.INPUT_OUTPUT.value, ["Input Output"], []),
            Element(FlowchartElementCategory.SUBROUTINE.value, ["Subroutine"], []),
            Element(FlowchartElementCategory.TERMINAL.value, ["Terminal state"], []),
        ]

        relations: List[Relation] = [
            Relation(FlowchartRelationCategory.ARROW.value, id_a, id_b, ["Arrow"], [], [], []),
            Relation(FlowchartRelationCategory.DOTTED_ARROW.value, id_b, id_c, ["Dotted Arrow"], [], [], []),
            Relation(FlowchartRelationCategory.ARROW.value, id_c, id_d, [], [], [], []),
            Relation(FlowchartRelationCategory.ARROW.value, id_d, id_e, [], [], [], []),
            Relation(FlowchartRelationCategory.OPEN_LINK.value, id_e, id_f, ["Open Link"], [], [], []),
        ]

        representation = FlowchartRepresentation(elements, relations)
        expected_result = TransducerOutcome("test_diagram", WellKnownMarkupLanguage.D2_LANG.value, "0: Circle\n"
                                                                                         "0.shape: circle\n"
                                                                                         "1: Process\n"
                                                                                         "1.style.border-radius: 8\n"
                                                                                         "2: Decision\n"
                                                                                         "2.shape: diamond\n"
                                                                                         "3: Input Output\n"
                                                                                         "3.shape: parallelogram\n"
                                                                                         "4: Subroutine\n"
                                                                                         "5: Terminal state\n"
                                                                                         "5.shape: oval\n"
                                                                                         "\n"
                                                                                         "0->1: Arrow\n"
                                                                                         "1->2: Dotted Arrow {\n"
                                                                                         "\tstyle: {\n"
                                                                                         "\t\tstroke-dash: 3\n"
                                                                                         "\t}\n"
                                                                                         "}"
                                                                                         "\n"
                                                                                         "2->3\n"
                                                                                         "3->4\n"
                                                                                         "4--5: Open Link\n")

        result = self.transducer.transduce("test_diagram", representation)
        self.assertEqual(expected_result.payload, result.payload)
