from typing import List, Type

from src.wellknown_diagram import WellKnownDiagram
from core.representation.representation import DiagramRepresentation
from src.representation.flowchart_representation.flowchart_representation import FlowchartRepresentation
from src.representation.flowchart_representation.element import FlowchartElementCategory, Element
from src.representation.flowchart_representation.relation import FlowchartRelationCategory, Relation
from core.transducer.outcome import TransducerOutcome
from core.transducer.transducer import Transducer
from src.wellknown_markuplang import WellKnownMarkupLanguage

import unittest
from typing import Dict


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

    @staticmethod
    def wrap_element(category: str, label: str, elem_id: str) -> str:
        match category:
            case FlowchartElementCategory.CIRCLE.value:
                return f"{elem_id}: {label}\n" \
                       f"{elem_id}.shape: circle\n"
            case FlowchartElementCategory.TERMINAL.value:
                return f"{elem_id}: {label}\n" \
                       f"{elem_id}.shape: oval\n"
            case FlowchartElementCategory.PROCESS.value:
                return f"{elem_id}: {label}\n" \
                       f"{elem_id}.style.border-radius: 8\n"
            case FlowchartElementCategory.DECISION.value:
                return f"{elem_id}: {label}\n" \
                       f"{elem_id}.shape: diamond\n"
            case FlowchartElementCategory.INPUT_OUTPUT.value:
                return f"{elem_id}: {label}\n" \
                       f"{elem_id}.shape: parallelogram\n"
            case FlowchartElementCategory.SUBROUTINE.value:
                return f"{elem_id}: {label}\n"
            case _:
                raise ValueError(f"Unknown flowchart element category: {category}")

    @staticmethod
    def wrap_relation(category: str, label: str, target_id: str) -> str:
        match category:
            case FlowchartRelationCategory.ARROW.value:
                if label == "":
                    return f"->{target_id}\n"
                return f"->{target_id}: {label}\n"
            case FlowchartRelationCategory.OPEN_LINK.value:
                if label == "":
                    return f"--{target_id}\n"
                return f"--{target_id}: {label}\n"
            case FlowchartRelationCategory.DOTTED_ARROW.value:
                if label == "":
                    return f"->{target_id} {{\n" \
                           "\tstyle: {\n" \
                           "\tstroke-dash: 3\n" \
                           "\t}\n" \
                           "}\n"
                return f"->{target_id}: {label} {{\n" \
                       "\tstyle: {\n" \
                       "\t\tstroke-dash: 3\n" \
                       "\t}\n" \
                       "}\n"
            case _:
                raise ValueError(f"Unknown flowchart relation category: {category}")

    def transduce(self, diagram_id: str, diagram_representation: DiagramRepresentation) -> TransducerOutcome:
        assert isinstance(diagram_representation, FlowchartRepresentation)

        body: str = ""
        for element in diagram_representation.elements.values():
            body += self.wrap_element(element.category, element.label, element.identifier)

        body += "\n"
        for relation in diagram_representation.relations:
            body += f"{relation.source_id}"
            body += self.wrap_relation(relation.category, relation.label, relation.target_id)

        outcome: TransducerOutcome = TransducerOutcome(diagram_id, WellKnownMarkupLanguage.D2_LANG.value, body)
        return outcome


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
            Relation(FlowchartRelationCategory.OPEN_LINK.value, id_e, id_f, "Open Link"),
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
                                                                                         "D->E\n"
                                                                                         "E--F: Open Link\n")

        result = self.transducer.transduce("test_diagram", representation)
        self.assertEqual(expected_result.payload, result.payload)
