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

    def test_transduce(self):
        id_a, id_b, id_c, id_d, id_e, id_f = "A", "B", "C", "D", "E", "F"

        elements: Dict[str, Element] = {
            id_a: Element(id_a, FlowchartElementCategory.CIRCLE.value, "Circle"),
            id_b: Element(id_b, FlowchartElementCategory.PROCESS.value, "Process"),
            id_c: Element(id_c, FlowchartElementCategory.DECISION.value, "Decision"),
            id_d: Element(id_d, FlowchartElementCategory.INPUT_OUTPUT.value, "Input Output"),
            id_e: Element(id_e, FlowchartElementCategory.SUBROUTINE.value, "Subroutine"),
            id_f: Element(id_f, FlowchartElementCategory.TERMINAL.value, "Terminal state"),
        }

        relations: List[Relation] = [
            Relation(FlowchartRelationCategory.ARROW.value, id_a, id_b, "Arrow"),
            Relation(FlowchartRelationCategory.DOTTED_ARROW.value, id_b, id_c, "Dotted Arrow"),
            Relation(FlowchartRelationCategory.ARROW.value, id_c, id_d, ""),
            Relation(FlowchartRelationCategory.ARROW.value, id_d, id_e, ""),
        ]

        representation = FlowchartRepresentation(elements, relations)
        expected_result = TransducerOutcome("test_diagram", WellKnownMarkupLanguage.D2_LANG.value, "A: Circle\n"
                                                                                         "A.shape: circle\n"
                                                                                         "B: Process\n"
                                                                                         "B.style.border-radius: 8\n"
                                                                                         "C: Decision\n"
                                                                                         "C.shape: diamond\n"
                                                                                         "D: Input Output\n"
                                                                                         "D.shape: parallelogram\n"
                                                                                         "E: Subroutine\n"
                                                                                         "F: Terminal state\n"
                                                                                         "F.shape: oval\n"
                                                                                         "\n"
                                                                                         "A->B: Arrow\n"
                                                                                         "B->C: Dotted Arrow {\n"
                                                                                         "\tstyle: {\n"
                                                                                         "\t\tstroke-dash: 3\n"
                                                                                         "\t}\n"
                                                                                         "}"
                                                                                         "\n"
                                                                                         "C->D\n"
                                                                                         "D->E\n")

        result = self.transducer.transduce("test_diagram", representation)
        self.assertEqual(expected_result.payload, result.payload)